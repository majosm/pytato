import numpy as np
import loopy as lp
from loopy.types import NumpyType
from typing import Dict, Optional
from numbers import Number
from pytato.array import (DictOfNamedArrays, Namespace, Array, ShapeType,
        NamedArray, normalize_shape)
from loopy.symbolic import IdentityMapper
from pytools import memoize_method
import pymbolic.primitives as prim


class ToPytatoSubstitutor(IdentityMapper):
    def __init__(self, bindings):
        self.bindings = bindings

    def map_product(self, expr):
        from functools import reduce
        import operator
        return reduce(operator.mul, expr.children, 1)

    def map_sum(self, expr):
        return sum(expr.children, start=0)

    def map_variable(self, expr):
        try:
            return self.bindings[expr.name]
        except KeyError:
            raise print(f"Got unknown variable '{expr.name}'")


class LoopyFunction(DictOfNamedArrays):
    """
    Call to a :mod:`loopy` program.
    """
    _mapper_method = "map_loopy_function"

    def __init__(self,
            namespace: Namespace,
            program: lp.Program,
            bindings: Dict[str, Array],
            entrypoint: str):
        super().__init__({})

        self.program = program
        self.bindings = bindings
        self.entrypoint = entrypoint
        self._namespace = namespace

        entry_kernel = program[entrypoint]

        self._named_arrays = {name: LoopyFunctionResult(self, name)
                              for name, lp_arg in entry_kernel.arg_dict.items()
                              if lp_arg.is_output}

    @memoize_method
    def to_pytato(self, expr):
        return ToPytatoSubstitutor(self.bindings)(expr)

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    @property
    def entry_kernel(self) -> lp.LoopKernel:
        return self.program[self.entrypoint]

    def __hash__(self):
        return hash((self.program, tuple(self.bindings.items()), self.entrypoint))

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, LoopyFunction):
            return False

        if ((self.entrypoint == other.entrypoint)
             and (self.bindings == other.bindings)
             and (self.program == other.program)):
            return True
        return False


class LoopyFunctionResult(NamedArray):
    """
    """
    def expr(self) -> Array:
        raise ValueError("Expressions for results of loopy functions aren't defined")

    @property
    def shape(self) -> ShapeType:
        loopy_arg = self.dict_of_named_arrays.entry_kernel.arg_dict[self.name]
        return self.dict_of_named_arrays.to_pytato(loopy_arg.shape)

    @property
    def dtype(self) -> np.dtype:
        loopy_arg = self.dict_of_named_arrays.entry_kernel.arg_dict[self.name]
        dtype = loopy_arg.dtype

        if isinstance(dtype, np.dtype):
            return dtype
        elif isinstance(dtype, NumpyType):
            return dtype.numpy_dtype
        else:
            raise NotImplementedError(f"Unknown dtype type '{dtype}'")


def call_loopy(namespace: Namespace, program: lp.Program, bindings: dict,
        entrypoint: Optional[str] = None):
    """
    Operates a general :class:`loopy.Program` on the array inputs as specified
    by *bindings*.

    Restrictions on the structure of ``program[entrypoint]``:

    * array arguments of ``program[entrypoint]`` should either be either
      input-only or output-only.
    * all input-only arguments of ``program[entrypoint]`` must appear in
      *bindings*.
    * all output-only arguments of ``program[entrypoint]`` must appear in
      *bindings*.
    * if *program* has been declared with multiple entrypoints, *entrypoint*
      can not be *None*.

    :arg bindings: mapping from argument names of ``program[entrypoint]`` to
        :class:`pytato.array.Array`.
    :arg results: names of ``program[entrypoint]`` argument names that have to
        be returned from the call.
    """
    if entrypoint is None:
        if len(program.entrypoints) != 1:
            raise ValueError("cannot infer entrypoint")

        entrypoint, = program.entrypoints

    program = program.with_entrypoints(entrypoint)

    # {{{ sanity checks

    if any([arg.is_input and arg.is_output
            for arg in program[entrypoint].args]):
        # Pytato DAG cannot have stateful nodes.
        raise ValueError("Cannot have a kernel that writes to inputs.")

    for arg in program[entrypoint].args:
        if arg.is_input:
            if arg.name not in bindings:
                raise ValueError(f"Kernel '{entrypoint}' expects an input"
                        f" '{arg.name}'")
            if isinstance(arg, lp.ArrayArg):
                if not isinstance(bindings[arg.name], Array):
                    raise ValueError(f"Argument '{arg.name}' expected to be a "
                            f"pytato.Array, got {type(bindings[arg.name])}.")
            else:
                assert isinstance(arg, lp.ValueArg)
                if not (isinstance(bindings[arg.name], (str,
                        prim.Expression, Number))):
                    raise ValueError(f"Argument '{arg.name}' expected to be a "
                            " number or a scalar expression, got "
                            f"{type(bindings[arg.name])}.")

                if isinstance(bindings[arg.name], str):
                    bindings[arg.name] = normalize_shape(bindings[arg.name],
                            namespace)[0]

    # }}}

    # {{{ infer types of the program

    for name, ary in bindings.items():
        if isinstance(ary, Array):
            program = lp.add_dtypes(program, {name: ary.dtype})
        elif isinstance(ary, prim.Expression):
            program = lp.add_dtypes(program, {name: np.intp})
        else:
            assert isinstance(arg, Number)
            program = lp.add_dtypes(program, {name, type(ary)})

    program = lp.infer_unknown_types(program)

    # }}}

    # {{{ infer shapes of the program

    program = lp.infer_arg_descr(program)

    # }}}

    program = program.with_entrypoints(frozenset())

    return LoopyFunction(namespace, program, bindings, entrypoint)


# vim: fdm=marker
