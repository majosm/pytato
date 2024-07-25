from __future__ import annotations

from pytools import memoize_method

__copyright__ = """
Copyright (C) 2020 Matt Wala
Copyright (C) 2020-21 Kaushik Kulkarni
Copyright (C) 2020-21 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import attrs
import logging
import numpy as np
from immutabledict import immutabledict
from typing import (Any, Callable, Dict, FrozenSet, Union, TypeVar, Set, Generic,
                    List, Mapping, Iterable, Tuple, Optional, TYPE_CHECKING,
                    Hashable, cast)

from pytato.array import (
        Array, IndexLambda, Placeholder, Stack, Roll,
        AxisPermutation, DataWrapper, SizeParam, DictOfNamedArrays,
        AbstractResultWithNamedArrays, Reshape, Concatenate, NamedArray,
        IndexRemappingBase, Einsum, InputArgumentBase,
        BasicIndex, AdvancedIndexInContiguousAxes, AdvancedIndexInNoncontiguousAxes,
        IndexBase, DataInterface)

from pytato.distributed.nodes import (
        DistributedSendRefHolder, DistributedRecv, DistributedSend)
from pytato.loopy import LoopyCall, LoopyCallResult
from pytato.function import Call, NamedCallResult, FunctionDefinition
from dataclasses import dataclass
from pytato.tags import ImplStored
from pymbolic.mapper.optimize import optimize_mapper


ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
MappedT = TypeVar("MappedT",
                  Array, AbstractResultWithNamedArrays, ArrayOrNames)
CombineT = TypeVar("CombineT")  # used in CombineMapper
TransformMapperResultT = TypeVar("TransformMapperResultT",  # used in TransformMapper
                            Array, AbstractResultWithNamedArrays, ArrayOrNames)
CopyMapperResultT = TypeVar("CopyMapperResultT",  # used in CopyMapper
                            Array, AbstractResultWithNamedArrays, ArrayOrNames)
CachedMapperT = TypeVar("CachedMapperT")  # used in CachedMapper
IndexOrShapeExpr = TypeVar("IndexOrShapeExpr")
R = FrozenSet[Array]
_SelfMapper = TypeVar("_SelfMapper", bound="Mapper")

__doc__ = """
.. currentmodule:: pytato.transform

.. autoclass:: Mapper
.. autoclass:: CachedMapper
.. autoclass:: TransformMapper
.. autoclass:: TransformMapperWithExtraArgs
.. autoclass:: CopyMapper
.. autoclass:: CopyMapperWithExtraArgs
.. autoclass:: CombineMapper
.. autoclass:: DependencyMapper
.. autoclass:: InputGatherer
.. autoclass:: SizeParamGatherer
.. autoclass:: SubsetDependencyMapper
.. autoclass:: WalkMapper
.. autoclass:: CachedWalkMapper
.. autoclass:: TopoSortMapper
.. autoclass:: CachedMapAndCopyMapper
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies
.. autofunction:: map_and_copy
.. autofunction:: materialize_with_mpms
.. autofunction:: deduplicate_data_wrappers
.. automodule:: pytato.transform.lower_to_index_lambda
.. automodule:: pytato.transform.remove_broadcasts_einsum
.. automodule:: pytato.transform.einsum_distributive_law
.. currentmodule:: pytato.transform

Dict representation of DAGs
---------------------------

.. autoclass:: UsersCollector
.. autofunction:: tag_user_nodes
.. autofunction:: rec_get_user_nodes


Transforming call sites
-----------------------

.. automodule:: pytato.transform.calls

.. currentmodule:: pytato.transform

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: MappedT

    A type variable representing the input type of a :class:`Mapper`.

.. class:: CombineT

    A type variable representing the type of a :class:`CombineMapper`.

.. class:: _SelfMapper

    A type variable used to represent the type of a mapper in
    :meth:`TransformMapper.clone_for_callee`.
"""

transform_logger = logging.getLogger(__file__)


class UnsupportedArrayError(ValueError):
    pass


# {{{ mapper base class

class Mapper:
    """A class that when called with a :class:`pytato.Array` recursively
    iterates over the DAG, calling the *_mapper_method* of each node. Users of
    this class are expected to override the methods of this class or create a
    subclass.

    .. note::

       This class might visit a node multiple times. Use a :class:`CachedMapper`
       if this is not desired.

    .. automethod:: handle_unsupported_array
    .. automethod:: map_foreign
    .. automethod:: rec
    .. automethod:: __call__
    """

    def handle_unsupported_array(self, expr: MappedT,
                                 *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError("%s cannot handle expressions of type %s"
                % (type(self).__name__, type(expr)))

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for an object of class for which a
        mapper method does not exist in this mapper.
        """
        raise ValueError("%s encountered invalid foreign object: %s"
                % (type(self).__name__, repr(expr)))

    def rec(self, expr: MappedT, *args: Any, **kwargs: Any) -> Any:
        """Call the mapper method of *expr* and return the result."""
        method: Optional[Callable[..., Array]]

        try:
            method = getattr(self, expr._mapper_method)
        except AttributeError:
            if isinstance(expr, Array):
                for cls in type(expr).__mro__[1:]:
                    method_name = getattr(cls, "_mapper_method", None)
                    if method_name:
                        method = getattr(self, method_name, None)
                        if method:
                            break
                else:
                    return self.handle_unsupported_array(expr, *args, **kwargs)
            else:
                return self.map_foreign(expr, *args, **kwargs)

        assert method is not None
        return method(expr, *args, **kwargs)

    def __call__(self, expr: MappedT, *args: Any, **kwargs: Any) -> Any:
        """Handle the mapping of *expr*."""
        return self.rec(expr, *args, **kwargs)

# }}}


# {{{ CachedMapper

class CachedMapper(Mapper, Generic[CachedMapperT]):
    """Mapper class that maps each node in the DAG exactly once. This loses some
    information compared to :class:`Mapper` as a node is visited only from
    one of its predecessors.

    .. automethod:: clone_for_callee
    """

    def __init__(self, err_on_collision: bool = True) -> None:
        super().__init__()
        self._err_on_collision = err_on_collision
        self._seen_exprs: Dict[Hashable, ArrayOrNames] = {}
        self._cache: Dict[Hashable, CachedMapperT] = {}

    def get_cache_key(self, expr: ArrayOrNames) -> Hashable:
        return expr

    def rec(self, expr: ArrayOrNames) -> CachedMapperT:
        key = self.get_cache_key(expr)
        try:
            result = self._cache[key]
        except KeyError:
            if self._err_on_collision:
                self._seen_exprs[key] = expr
            result = super().rec(expr)
            self._cache[key] = result
        else:
            if self._err_on_collision and expr is not self._seen_exprs[key]:
                raise ValueError(
                    f"cache collision detected on {type(expr)} in {type(self)}.")

        return result  # type: ignore[return-value]

    if TYPE_CHECKING:
        def __call__(self, expr: ArrayOrNames) -> CachedMapperT:
            return self.rec(expr)

    @memoize_method
    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        return type(self)(err_on_collision=self._err_on_collision)

# }}}


# {{{ TransformMapper

class TransformMapper(CachedMapper[ArrayOrNames]):
    """Base class for mappers that transform :class:`pytato.array.Array`\\ s into
    other :class:`pytato.array.Array`\\ s.

    .. automethod:: clone_for_callee
    """
    if TYPE_CHECKING:
        def __call__(self, expr: TransformMapperResultT) -> TransformMapperResultT:
            return cast(TransformMapperResultT, super().rec(expr))

    def __init__(
            self,
            err_on_collision: bool = True,
            err_on_duplication: Optional[bool] = None) -> None:
        """
        :arg err_on_collision: Raise an exception if two distinct input array
            instances have the same key.
        :arg err_on_duplication: Raise an exception if mapping produces a new array
            instance that has the same key as the input array. Requires
            `err_on_collision=True`. Defaults to *True* if `err_on_collision` is
            enabled.
        """
        super().__init__(err_on_collision=err_on_collision)
        if err_on_duplication is None:
            err_on_duplication = err_on_collision
        if err_on_duplication and not err_on_collision:
            raise ValueError(
                "err_on_duplication=True requires err_on_collision=True.")
        self._err_on_duplication = err_on_duplication
        self._seen_results: Dict[Hashable, TransformMapperResultT] = {}

    def rec(self, expr: TransformMapperResultT) -> TransformMapperResultT:
        key = self.get_cache_key(expr)
        try:
            result = self._cache[key]
        except KeyError:
            pass
        else:
            if self._err_on_collision and expr is not self._seen_exprs[key]:
                raise ValueError(
                    f"cache collision detected on {type(expr)} in {type(self)}.")
            return result

        if self._err_on_collision:
            self._seen_exprs[key] = expr

        result = Mapper.rec(self, expr)
        result_key = self.get_cache_key(result)

        # This only works if the expression has no existing duplicates (hence
        # the err_on_collision=True requirement). Otherwise, rec() could produce a
        # valid result that is not identical to expr due to deduplication
        if (
                self._err_on_duplication
                and hash(result_key) == hash(key)
                and result_key == key
                and result is not expr):
            raise ValueError(
                f"array duplication detected on {type(expr)} in {type(self)}.")

        try:
            result = self._seen_results[result_key]
        except KeyError:
            self._seen_results[result_key] = result

        self._cache[key] = result

        return result  # type: ignore[return-value]

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        return type(self)(
            err_on_collision=self._err_on_collision,
            err_on_duplication=self._err_on_duplication)

# }}}


# {{{ TransformMapperWithExtraArgs

class TransformMapperWithExtraArgs(CachedMapper[ArrayOrNames]):
    """
    Similar to :class:`TransformMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`TransformMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.

    .. automethod:: clone_for_callee
    """
    if TYPE_CHECKING:
        def __call__(
                self, expr: TransformMapperResultT, *args: Any, **kwargs: Any
                ) -> TransformMapperResultT:
            return self.rec(expr, *args, **kwargs)

    def __init__(
            self,
            err_on_collision: bool = True,
            err_on_duplication: Optional[bool] = None) -> None:
        """
        :arg err_on_collision: Raise an exception if two distinct input array
        instances have the same key.
        :arg err_on_duplication: Raise an exception if mapping produces a new array
            instance that has the same key as the input array. Requires
            `err_on_collision=True`. Defaults to *True* if `err_on_collision` is
            enabled.
        """
        super().__init__(err_on_collision)
        # type-ignored as '._cache' attribute is not coherent with the base
        # class
        self._cache: Dict[Tuple[ArrayOrNames,
                                Tuple[Any, ...],
                                Tuple[Tuple[str, Any], ...]
                                ],
                          ArrayOrNames] = {}  # type: ignore[assignment]
        if err_on_duplication is None:
            err_on_duplication = err_on_collision
        if err_on_duplication and not err_on_collision:
            raise ValueError(
                "err_on_duplication=True requires err_on_collision=True.")
        self._err_on_duplication = err_on_duplication
        self._seen_results: Dict[Hashable, TransformMapperResultT] = {}

    def get_cache_key(self,
                      expr: ArrayOrNames,
                      *args: Any, **kwargs: Any
                      ) -> Union[
                        ArrayOrNames,
                        Tuple[
                            ArrayOrNames,
                            Tuple[Any, ...],
                            Tuple[Tuple[str, Any], ...]]]:
        extras = []
        if args:
            extras.append(args)
        if kwargs:
            extras.append(tuple(sorted(kwargs.items())))

        if extras:
            return (expr, *extras)
        else:
            return expr

    def rec(self,
            expr: TransformMapperResultT,
            *args: Any, **kwargs: Any) -> TransformMapperResultT:
        key = self.get_cache_key(expr, *args, **kwargs)
        try:
            result = self._cache[key]
        except KeyError:
            pass
        else:
            if self._err_on_collision and expr is not self._seen_exprs[key]:
                raise ValueError(
                    f"cache collision detected on {type(expr)} in {type(self)}.")
            return result

        if self._err_on_collision:
            self._seen_exprs[key] = expr

        result = Mapper.rec(self, expr, *args, **kwargs)
        result_key = self.get_cache_key(result)

        # This only works if the expression has no existing duplicates (hence
        # the err_on_collision=True requirement). Otherwise, rec() could produce a
        # valid result that is not identical to expr due to deduplication
        if (
                self._err_on_duplication
                and hash(result_key) == hash(key)
                and result_key == key
                and result is not expr):
            raise ValueError(
                f"array duplication detected on {type(expr)} in {type(self)}.")

        try:
            result = self._seen_results[result_key]
        except KeyError:
            self._seen_results[result_key] = result

        self._cache[key] = result

        return result  # type: ignore[return-value]

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        return type(self)(
            err_on_collision=self._err_on_collision,
            err_on_duplication=self._err_on_duplication)

# }}}


# {{{ CopyMapper

class CopyMapper(TransformMapper):
    """Performs a deep copy of a :class:`pytato.array.Array`.
    The typical use of this mapper is to override individual ``map_`` methods
    in subclasses to permit term rewriting on an expression graph.

    .. note::

       This does not copy the data of a :class:`pytato.array.DataWrapper`.
    """
    if TYPE_CHECKING:
        def rec(self, expr: CopyMapperResultT) -> CopyMapperResultT:
            return cast(CopyMapperResultT, super().rec(expr))

        def __call__(self, expr: CopyMapperResultT) -> CopyMapperResultT:
            return self.rec(expr)

    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...]
                              ) -> Tuple[IndexOrShapeExpr, ...]:
        # type-ignore-reason: apparently mypy cannot substitute typevars
        # here.
        new_situp = tuple(
            self.rec(s) if isinstance(s, Array) else s  # type: ignore[misc]
            for s in situp)
        if all(new_s is s for s, new_s in zip(situp, new_situp)):
            return situp
        else:
            return new_situp

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        new_bindings: Mapping[str, Array] = immutabledict({
                name: self.rec(subexpr)
                for name, subexpr in sorted(expr.bindings.items())})
        if (
                new_shape is expr.shape
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return IndexLambda(expr=expr.expr,
                    shape=new_shape,
                    dtype=expr.dtype,
                    bindings=new_bindings,
                    axes=expr.axes,
                    var_to_reduction_descr=expr.var_to_reduction_descr,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        assert expr.name is not None
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        if new_shape is expr.shape:
            return expr
        else:
            return Placeholder(name=expr.name,
                    shape=new_shape,
                    dtype=expr.dtype,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_stack(self, expr: Stack) -> Array:
        new_arrays = tuple(self.rec(arr) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Stack(arrays=new_arrays, axis=expr.axis, axes=expr.axes,
                    tags=expr.tags, non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays = tuple(self.rec(arr) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Concatenate(arrays=new_arrays, axis=expr.axis,
                               axes=expr.axes, tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll) -> Array:
        new_ary = self.rec(expr.array)
        if new_ary is expr.array:
            return expr
        else:
            return Roll(array=new_ary,
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_ary = self.rec(expr.array)
        if new_ary is expr.array:
            return expr
        else:
            return AxisPermutation(array=new_ary,
                    axis_permutation=expr.axis_permutation,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        new_ary = self.rec(expr.array)
        new_indices = self.rec_idx_or_size_tuple(expr.indices)
        if new_ary is expr.array and new_indices is expr.indices:
            return expr
        else:
            return type(expr)(new_ary,
                              indices=new_indices,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_basic_index(self, expr: BasicIndex) -> Array:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> Array:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> Array:
        return self._map_index_base(expr)

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        if new_shape is expr.shape:
            return expr
        else:
            return DataWrapper(
                    data=expr.data,
                    shape=new_shape,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        assert expr.name is not None
        return expr

    def map_einsum(self, expr: Einsum) -> Array:
        new_args = tuple(self.rec(arg) for arg in expr.args)
        if all(new_arg is arg for arg, new_arg in zip(expr.args, new_args)):
            return expr
        else:
            return Einsum(expr.access_descriptors,
                          new_args,
                          axes=expr.axes,
                          redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                          index_to_access_descr=expr.index_to_access_descr,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_named_array(self, expr: NamedArray) -> Array:
        new_container = self.rec(expr._container)
        if new_container is expr._container:
            return expr
        else:
            return type(expr)(new_container,
                              expr.name,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays) -> DictOfNamedArrays:
        new_data = {
            key: self.rec(val.expr)
            for key, val in expr.items()}
        if all(
                new_data_val is val.expr
                for val, new_data_val in zip(expr.values(), new_data.values())):
            return expr
        else:
            return DictOfNamedArrays(new_data, tags=expr.tags)

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        new_bindings: Mapping[Any, Any] = immutabledict(
                    {name: (self.rec(subexpr) if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})
        if all(
                new_bnd is bnd
                for bnd, new_bnd in zip(
                    expr.bindings.values(),
                    new_bindings.values())):
            return expr
        else:
            return LoopyCall(translation_unit=expr.translation_unit,
                             bindings=new_bindings,
                             entrypoint=expr.entrypoint,
                             tags=expr.tags,
                             )

    def map_loopy_call_result(self, expr: LoopyCallResult) -> Array:
        new_container = self.rec(expr._container)
        assert isinstance(new_container, LoopyCall)
        if new_container is expr._container:
            return expr
        else:
            return LoopyCallResult(
                    container=new_container,
                    name=expr.name,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        new_ary = self.rec(expr.array)
        new_newshape = self.rec_idx_or_size_tuple(expr.newshape)
        if new_ary is expr.array and new_newshape is expr.newshape:
            return expr
        else:
            return Reshape(new_ary,
                           newshape=new_newshape,
                           order=expr.order,
                           axes=expr.axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        new_send_data = self.rec(expr.send.data)
        if new_send_data is expr.send.data:
            new_send = expr.send
        else:
            new_send = DistributedSend(
                data=new_send_data,
                dest_rank=expr.send.dest_rank,
                comm_tag=expr.send.comm_tag)
        new_passthrough = self.rec(expr.passthrough_data)
        if new_send is expr.send and new_passthrough is expr.passthrough_data:
            return expr
        else:
            return DistributedSendRefHolder(
                    new_send,
                    new_passthrough,
                    axes=new_passthrough.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_distributed_recv(self, expr: DistributedRecv) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        if new_shape is expr.shape:
            return expr
        else:
            return DistributedRecv(
                   src_rank=expr.src_rank, comm_tag=expr.comm_tag,
                   shape=new_shape, dtype=expr.dtype, tags=expr.tags,
                   axes=expr.axes, non_equality_tags=expr.non_equality_tags)

    @memoize_method
    def map_function_definition(self,
                                expr: FunctionDefinition) -> FunctionDefinition:
        # spawn a new mapper to avoid unsound cache hits, since the namespace of the
        # function's body is different from that of the caller.
        new_mapper = self.clone_for_callee(expr)
        new_returns = {name: new_mapper(ret)
                       for name, ret in expr.returns.items()}
        if all(
                new_ret is ret
                for ret, new_ret in zip(
                    expr.returns.values(),
                    new_returns.values())):
            return expr
        else:
            return attrs.evolve(expr, returns=immutabledict(new_returns))

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        new_function = self.map_function_definition(expr.function)
        new_bindings = {
            name: self.rec(bnd)
            for name, bnd in expr.bindings.items()}
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        new_call = self.rec(expr._container)
        assert isinstance(new_call, Call)
        return new_call[expr.name]


class CopyMapperWithExtraArgs(TransformMapperWithExtraArgs):
    """
    Similar to :class:`CopyMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`CopyMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.
    """
    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...],
                              *args: Any, **kwargs: Any
                              ) -> Tuple[IndexOrShapeExpr, ...]:
        # type-ignore-reason: apparently mypy cannot substitute typevars
        # here.
        return tuple(
            self.rec(s, *args, **kwargs)  # type: ignore[misc]
            if isinstance(s, Array)
            else s
            for s in situp)

    def map_index_lambda(self, expr: IndexLambda,
                         *args: Any, **kwargs: Any) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        new_bindings: Mapping[str, Array] = immutabledict({
                name: self.rec(subexpr, *args, **kwargs)
                for name, subexpr in sorted(expr.bindings.items())})
        if (
                new_shape is expr.shape
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return IndexLambda(expr=expr.expr,
                               shape=new_shape,
                               dtype=expr.dtype,
                               bindings=new_bindings,
                               axes=expr.axes,
                               var_to_reduction_descr=expr.var_to_reduction_descr,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder, *args: Any, **kwargs: Any) -> Array:
        assert expr.name is not None
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        if new_shape is expr.shape:
            return expr
        else:
            return Placeholder(name=expr.name,
                               shape=new_shape,
                               dtype=expr.dtype,
                               axes=expr.axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_stack(self, expr: Stack, *args: Any, **kwargs: Any) -> Array:
        new_arrays = tuple(self.rec(arr, *args, **kwargs) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Stack(arrays=new_arrays, axis=expr.axis, axes=expr.axes,
                    tags=expr.tags, non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate, *args: Any, **kwargs: Any) -> Array:
        new_arrays = tuple(self.rec(arr, *args, **kwargs) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Concatenate(arrays=new_arrays, axis=expr.axis,
                               axes=expr.axes, tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll, *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        if new_ary is expr.array:
            return expr
        else:
            return Roll(array=new_ary,
                        shift=expr.shift,
                        axis=expr.axis,
                        axes=expr.axes,
                        tags=expr.tags,
                        non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation,
                             *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        if new_ary is expr.array:
            return expr
        else:
            return AxisPermutation(array=new_ary,
                                   axis_permutation=expr.axis_permutation,
                                   axes=expr.axes,
                                   tags=expr.tags,
                                   non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase, *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        new_indices = self.rec_idx_or_size_tuple(expr.indices, *args, **kwargs)
        if new_ary is expr.array and new_indices is expr.indices:
            return expr
        else:
            return type(expr)(new_ary,
                              indices=new_indices,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_basic_index(self, expr: BasicIndex, *args: Any, **kwargs: Any) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      *args: Any, **kwargs: Any

                                      ) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          *args: Any, **kwargs: Any
                                          ) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_data_wrapper(self, expr: DataWrapper,
                         *args: Any, **kwargs: Any) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        if new_shape is expr.shape:
            return expr
        else:
            return DataWrapper(
                    data=expr.data,
                    shape=new_shape,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_size_param(self, expr: SizeParam, *args: Any, **kwargs: Any) -> Array:
        assert expr.name is not None
        return expr

    def map_einsum(self, expr: Einsum, *args: Any, **kwargs: Any) -> Array:
        new_args = tuple(self.rec(arg, *args, **kwargs) for arg in expr.args)
        if all(new_arg is arg for arg, new_arg in zip(expr.args, new_args)):
            return expr
        else:
            return Einsum(expr.access_descriptors,
                          new_args,
                          axes=expr.axes,
                          redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                          index_to_access_descr=expr.index_to_access_descr,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_named_array(self, expr: NamedArray, *args: Any, **kwargs: Any) -> Array:
        new_container = self.rec(expr._container, *args, **kwargs)
        if new_container is expr._container:
            return expr
        else:
            return type(expr)(new_container,
                              expr.name,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays, *args: Any, **kwargs: Any) -> DictOfNamedArrays:
        new_data = {
            key: self.rec(val.expr, *args, **kwargs)
            for key, val in expr.items()}
        if all(
                new_data_val is val.expr
                for val, new_data_val in zip(expr.values(), new_data.values())):
            return expr
        else:
            return DictOfNamedArrays(new_data, tags=expr.tags)

    def map_loopy_call(self, expr: LoopyCall,
                       *args: Any, **kwargs: Any) -> LoopyCall:
        new_bindings: Mapping[Any, Any] = immutabledict(
                    {name: (self.rec(subexpr, *args, **kwargs)
                           if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})
        if all(
                new_bnd is bnd
                for bnd, new_bnd in zip(
                    expr.bindings.values(),
                    new_bindings.values())):
            return expr
        else:
            return LoopyCall(translation_unit=expr.translation_unit,
                             bindings=new_bindings,
                             entrypoint=expr.entrypoint,
                             tags=expr.tags,
                             )

    def map_loopy_call_result(self, expr: LoopyCallResult,
                              *args: Any, **kwargs: Any) -> Array:
        new_container = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_container, LoopyCall)
        if new_container is expr._container:
            return expr
        else:
            return LoopyCallResult(
                    container=new_container,
                    name=expr.name,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape,
                    *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        new_newshape = self.rec_idx_or_size_tuple(expr.newshape, *args, **kwargs)
        if new_ary is expr.array and new_newshape is expr.newshape:
            return expr
        else:
            return Reshape(new_ary,
                           newshape=new_newshape,
                           order=expr.order,
                           axes=expr.axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder,
                                        *args: Any, **kwargs: Any) -> Array:
        new_send_data = self.rec(expr.send.data, *args, **kwargs)
        if new_send_data is expr.send.data:
            new_send = expr.send
        else:
            new_send = DistributedSend(
                data=new_send_data,
                dest_rank=expr.send.dest_rank,
                comm_tag=expr.send.comm_tag)
        new_passthrough = self.rec(expr.passthrough_data, *args, **kwargs)
        if new_send is expr.send and new_passthrough is expr.passthrough_data:
            return expr
        else:
            return DistributedSendRefHolder(
                    new_send,
                    new_passthrough,
                    axes=new_passthrough.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: Any, **kwargs: Any) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        if new_shape is expr.shape:
            return expr
        else:
            return DistributedRecv(
                   src_rank=expr.src_rank, comm_tag=expr.comm_tag,
                   shape=new_shape, dtype=expr.dtype, tags=expr.tags,
                   axes=expr.axes, non_equality_tags=expr.non_equality_tags)

    def map_function_definition(self, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> FunctionDefinition:
        raise NotImplementedError("Function definitions are purposefully left"
                                  " unimplemented as the default arguments to a new"
                                  " DAG traversal are tricky to guess.")

    def map_call(self, expr: Call,
                 *args: Any, **kwargs: Any) -> AbstractResultWithNamedArrays:
        new_function = self.map_function_definition(expr.function, *args, **kwargs)
        new_bindings = {
            name: self.rec(bnd, *args, **kwargs)
            for name, bnd in expr.bindings.items()}
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: Any, **kwargs: Any) -> Array:
        new_call = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_call, Call)
        return new_call[expr.name]

# }}}


# {{{ CombineMapper

class CombineMapper(Mapper, Generic[CombineT]):
    """
    Abstract mapper that recursively combines the results of user nodes
    of a given expression.

    .. automethod:: combine
    """
    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[ArrayOrNames, CombineT] = {}

    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...]
                              ) -> Tuple[CombineT, ...]:
        return tuple(self.rec(s) for s in situp if isinstance(s, Array))

    def rec(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: CombineT = super().rec(expr)
        self.cache[expr] = result
        return result

    # type-ignore reason: incompatible ret. type with super class
    def __call__(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        return self.rec(expr)

    def combine(self, *args: CombineT) -> CombineT:
        """Combine the arguments."""
        raise NotImplementedError

    def map_index_lambda(self, expr: IndexLambda) -> CombineT:
        return self.combine(*(self.rec(bnd)
                              for _, bnd in sorted(expr.bindings.items())),
                            *self.rec_idx_or_size_tuple(expr.shape))

    def map_placeholder(self, expr: Placeholder) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_data_wrapper(self, expr: DataWrapper) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_stack(self, expr: Stack) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.arrays))

    def map_roll(self, expr: Roll) -> CombineT:
        return self.combine(self.rec(expr.array))

    def map_axis_permutation(self, expr: AxisPermutation) -> CombineT:
        return self.combine(self.rec(expr.array))

    def _map_index_base(self, expr: IndexBase) -> CombineT:
        return self.combine(self.rec(expr.array),
                            *self.rec_idx_or_size_tuple(expr.indices))

    def map_basic_index(self, expr: BasicIndex) -> CombineT:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> CombineT:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> CombineT:
        return self._map_index_base(expr)

    def map_reshape(self, expr: Reshape) -> CombineT:
        return self.combine(
                self.rec(expr.array),
                *self.rec_idx_or_size_tuple(expr.newshape))

    def map_concatenate(self, expr: Concatenate) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.arrays))

    def map_einsum(self, expr: Einsum) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.args))

    def map_named_array(self, expr: NamedArray) -> CombineT:
        return self.combine(self.rec(expr._container))

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> CombineT:
        return self.combine(*(self.rec(ary.expr)
                              for ary in expr.values()))

    def map_loopy_call(self, expr: LoopyCall) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for _, ary in sorted(expr.bindings.items())
                              if isinstance(ary, Array)))

    def map_loopy_call_result(self, expr: LoopyCallResult) -> CombineT:
        return self.rec(expr._container)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> CombineT:
        return self.combine(
                self.rec(expr.send.data),
                self.rec(expr.passthrough_data),
                )

    def map_distributed_recv(self, expr: DistributedRecv) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition) -> CombineT:
        raise NotImplementedError("Combining results from a callee expression"
                                  " is context-dependent. Derived classes"
                                  " must override map_function_definition.")

    def map_call(self, expr: Call) -> CombineT:
        raise NotImplementedError(
            "Mapping calls is context-dependent. Derived classes must override "
            "map_call.")

    def map_named_call_result(self, expr: NamedCallResult) -> CombineT:
        return self.rec(expr._container)

# }}}


# {{{ DependencyMapper

class DependencyMapper(CombineMapper[R]):
    """
    Maps a :class:`pytato.array.Array` to a :class:`frozenset` of
    :class:`pytato.array.Array`'s it depends on.

    .. warning::

       This returns every node in the graph! Consider a custom
       :class:`CombineMapper` or a :class:`SubsetDependencyMapper` instead.
    """

    def combine(self, *args: R) -> R:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_index_lambda(self, expr: IndexLambda) -> R:
        return self.combine(frozenset([expr]), super().map_index_lambda(expr))

    def map_placeholder(self, expr: Placeholder) -> R:
        return self.combine(frozenset([expr]), super().map_placeholder(expr))

    def map_data_wrapper(self, expr: DataWrapper) -> R:
        return self.combine(frozenset([expr]), super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> R:
        return frozenset([expr])

    def map_stack(self, expr: Stack) -> R:
        return self.combine(frozenset([expr]), super().map_stack(expr))

    def map_roll(self, expr: Roll) -> R:
        return self.combine(frozenset([expr]), super().map_roll(expr))

    def map_axis_permutation(self, expr: AxisPermutation) -> R:
        return self.combine(frozenset([expr]), super().map_axis_permutation(expr))

    def _map_index_base(self, expr: IndexBase) -> R:
        return self.combine(frozenset([expr]), super()._map_index_base(expr))

    def map_reshape(self, expr: Reshape) -> R:
        return self.combine(frozenset([expr]), super().map_reshape(expr))

    def map_concatenate(self, expr: Concatenate) -> R:
        return self.combine(frozenset([expr]), super().map_concatenate(expr))

    def map_einsum(self, expr: Einsum) -> R:
        return self.combine(frozenset([expr]), super().map_einsum(expr))

    def map_named_array(self, expr: NamedArray) -> R:
        return self.combine(frozenset([expr]), super().map_named_array(expr))

    def map_loopy_call_result(self, expr: LoopyCallResult) -> R:
        return self.combine(frozenset([expr]), super().map_loopy_call_result(expr))

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> R:
        return self.combine(
                frozenset([expr]), super().map_distributed_send_ref_holder(expr))

    def map_distributed_recv(self, expr: DistributedRecv) -> R:
        return self.combine(frozenset([expr]), super().map_distributed_recv(expr))

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition) -> R:
        # do not include arrays from the function's body as it would involve
        # putting arrays from different namespaces into the same collection.
        return frozenset()

    def map_call(self, expr: Call) -> R:
        return self.combine(self.map_function_definition(expr.function),
                            *[self.rec(bnd) for bnd in expr.bindings.values()])

    def map_named_call_result(self, expr: NamedCallResult) -> R:
        return self.rec(expr._container)

# }}}


# {{{ SubsetDependencyMapper

class SubsetDependencyMapper(DependencyMapper):
    """
    Mapper to combine the dependencies of an expression that are a subset of
    *universe*.
    """
    def __init__(self, universe: FrozenSet[Array]):
        self.universe = universe
        super().__init__()

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(lambda acc, arg: acc | (arg & self.universe),
                      args,
                      frozenset())

# }}}


# {{{ InputGatherer

class InputGatherer(CombineMapper[FrozenSet[InputArgumentBase]]):
    """
    Mapper to combine all instances of :class:`pytato.array.InputArgumentBase` that
    an array expression depends on.
    """
    def combine(self, *args: FrozenSet[InputArgumentBase]
                ) -> FrozenSet[InputArgumentBase]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_placeholder(self, expr: Placeholder) -> FrozenSet[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_placeholder(expr))

    def map_data_wrapper(self, expr: DataWrapper) -> FrozenSet[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> FrozenSet[SizeParam]:
        return frozenset([expr])

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition
                                ) -> FrozenSet[InputArgumentBase]:
        # get rid of placeholders local to the function.
        new_mapper = InputGatherer()
        all_callee_inputs = new_mapper.combine(*[new_mapper(ret)
                                                 for ret in expr.returns.values()])
        result: Set[InputArgumentBase] = set()
        for inp in all_callee_inputs:
            if isinstance(inp, Placeholder):
                if inp.name in expr.parameters:
                    # drop, reference to argument
                    pass
                else:
                    raise ValueError("function definition refers to non-argument "
                                     f"placeholder named '{inp.name}'")
            else:
                result.add(inp)

        return frozenset(result)

    def map_call(self, expr: Call) -> FrozenSet[InputArgumentBase]:
        return self.combine(self.map_function_definition(expr.function),
            *[
                self.rec(bnd)
                for name, bnd in sorted(expr.bindings.items())])

# }}}


# {{{ precompute_subexpressions

# FIXME: Think about what happens when subexpressions contain outlined functions
class _PrecomputableSubexpressionGatherer(CombineMapper[FrozenSet[Array]]):
    """
    Mapper to find subexpressions that do not depend on any placeholders.
    """
    def rec(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: CombineT = super().rec(expr)
        if not isinstance(expr, (
                Placeholder,
                DictOfNamedArrays,
                Call)):
            from pytato.analysis import DirectPredecessorsGetter
            if result == DirectPredecessorsGetter()(expr):
                result = frozenset({expr})
        self.cache[expr] = result
        return result

    # type-ignore reason: incompatible ret. type with super class
    def __call__(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        subexprs = self.rec(expr)

        # Need to treat data arrays as precomputable during recursion, but afterwards
        # we only care about larger expressions containing them *or* their shape if
        # it's a non-constant expression
        # FIXME: Does it even make sense for a data array to have an expression as
        # a shape? Maybe this isn't necessary...

        data_subexprs = {
            ary
            for ary in subexprs
            if isinstance(ary, (DataWrapper, DistributedRecv))}

        subexprs -= data_subexprs

        for ary in data_subexprs:
            subexprs |= self.combine(*self.rec_idx_or_size_tuple(ary.shape))

        return subexprs

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition) -> CombineT:
        # FIXME: Ignoring subexpressions inside function definitions for now
        return frozenset()

    def map_call(self, expr: Call) -> CombineT:
        rec_fn = self.map_function_definition(expr.function)
        assert not rec_fn
        rec_bindings = immutabledict({
            name: self.rec(bnd) if isinstance(bnd, Array) else frozenset({bnd})
            for name, bnd in expr.bindings.items()})
        if all(
                rec_bindings[name] == frozenset({expr.bindings[name]})
                for name in expr.bindings):
            return frozenset({expr})
        else:
            return self.combine(rec_fn, *rec_bindings.values())


class _PrecomputableSubexpressionReplacer(CopyMapper):
    """
    Mapper to replace precomputable subexpressions found by
    :class:`_PrecomputableSubexpressionGatherer` with the evaluated versions.
    """
    def __init__(self, replacement_map: Mapping[Array, Array]) -> None:
        super().__init__()
        self.replacement_map = replacement_map

    # FIXME: It's awkward to have to duplicate all of this from TransformMapper;
    # figure out a better way
    def rec(self, expr: TransformMapperResultT) -> TransformMapperResultT:
        key = self.get_cache_key(expr)
        try:
            result = self._cache[key]
        except KeyError:
            pass
        else:
            if self._err_on_collision and expr is not self._seen_exprs[key]:
                raise ValueError(
                    f"cache collision detected on {type(expr)} in {type(self)}.")
            return result  # type: ignore[return-value]

        if self._err_on_collision:
            self._seen_exprs[key] = expr

        result = self.replacement_map.get(expr, None)
        if result is not None:
            result = self.rec(result)
        else:
            result = Mapper.rec(self, expr)

        result_key = self.get_cache_key(result)

        # This only works if the expression has no existing duplicates (hence
        # the err_on_collision=True requirement). Otherwise, rec() could produce a
        # valid result that is not identical to expr due to deduplication
        if (
                self._err_on_duplication
                and hash(result_key) == hash(key)
                and result_key == key
                and result is not expr):
            raise ValueError(
                f"array duplication detected on {type(expr)} in {type(self)}."
                f"{id(result)=}, {id(expr)=}")

        try:
            result = self._seen_results[result_key]
        except KeyError:
            self._seen_results[result_key] = result

        self._cache[key] = result

        return result  # type: ignore[return-value]

    @memoize_method
    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        # FIXME: Ignoring subexpressions inside function definitions for now
        return type(self)({})


def precompute_subexpressions(
        expr: ArrayOrNames,
        eval_func: Callable[ArrayOrNames]) -> ArrayOrNames:
    """Evaluate subexpressions in *expr* that do not depend on any placeholders."""
    precomputable_subexprs = _PrecomputableSubexpressionGatherer()(expr)
    for subexpr in precomputable_subexprs:
        from pytato.analysis import get_num_nodes
        nnodes = get_num_nodes(subexpr)
        if nnodes > 1:
            print(
                "Found precomputable subexpression of type "
                f"{type(subexpr).__name__} with {nnodes} nodes.")
    # FIXME: Assemble into DictOfNamedArrays and evaluate all in one go? Might be a
    # lot of overhead otherwise
    cpm = CopyMapper(err_on_collision=False)
    subexpr_to_evaled_subexpr = {
        subexpr: cpm(eval_func(subexpr))
        for subexpr in precomputable_subexprs}
    return _PrecomputableSubexpressionReplacer(subexpr_to_evaled_subexpr)(expr)
    # from pytato.array import make_dict_of_named_arrays
    # precomputable_subexprs_dict = make_dict_of_named_arrays({
    #     f"_{i}": subexpr
    #     for i, subexpr in enumerate(precomputable_subexprs)})
    # evaled_subexprs_dict = eval_func(precomputable_subexprs_dict)
    # subexpr_to_evaled_subexpr = {
    #     subexpr: evaled_subexprs_dict._data[f"_{i}"]
    #     for i, subexpr in enumerate(precomputable_subexprs)}
    # return _PrecomputableSubexpressionReplacer(subexpr_to_evaled_subexpr)(expr)

# }}}


# {{{ SelfComputeGatherer

class SelfComputeGatherer(CombineMapper[FrozenSet[Array]]):
    """
    Mapper to combine all non-materialized arrays that an array expression depends
    on.
    """
    def rec(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        if expr.tags_of_type(ImplStored):
            result: CombineT = frozenset()
        else:
            result: CombineT = self.combine(frozenset({expr}), Mapper.rec(self, expr))
        self.cache[expr] = result
        return result

    def combine(self, *args: FrozenSet[InputArgumentBase]
                ) -> FrozenSet[InputArgumentBase]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

# }}}


# {{{ SizeParamGatherer

class SizeParamGatherer(CombineMapper[FrozenSet[SizeParam]]):
    """
    Mapper to combine all instances of :class:`pytato.array.SizeParam` that
    an array expression depends on.
    """
    def combine(self, *args: FrozenSet[SizeParam]
                ) -> FrozenSet[SizeParam]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_size_param(self, expr: SizeParam) -> FrozenSet[SizeParam]:
        return frozenset([expr])

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition
                                ) -> FrozenSet[SizeParam]:
        return self.combine(*[self.rec(ret)
                              for ret in expr.returns.values()])

# }}}


# {{{ WalkMapper

class WalkMapper(Mapper):
    """
    A mapper that walks over all the arrays in a :class:`pytato.Array`.

    Users may override the specific mapper methods in a derived class or
    override :meth:`WalkMapper.visit` and :meth:`WalkMapper.post_visit`.

    .. automethod:: visit
    .. automethod:: post_visit
    """

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        return type(self)()

    def visit(self, expr: Any, *args: Any, **kwargs: Any) -> bool:
        """
        If this method returns *True*, *expr* is traversed during the walk.
        If this method returns *False*, *expr* is not traversed as a part of
        the walk.
        """
        return True

    def post_visit(self, expr: Any, *args: Any, **kwargs: Any) -> None:
        """
        Callback after *expr* has been traversed.
        """
        pass

    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...],
                              *args: Any, **kwargs: Any) -> None:
        for comp in situp:
            if isinstance(comp, Array):
                self.rec(comp, *args, **kwargs)

    def map_index_lambda(self, expr: IndexLambda, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for _, child in sorted(expr.bindings.items()):
            self.rec(child, *args, **kwargs)

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_placeholder(self, expr: Placeholder, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_idx_or_size_tuple(expr.shape)

        self.post_visit(expr, *args, **kwargs)

    map_data_wrapper = map_placeholder
    map_size_param = map_placeholder

    def _map_index_remapping_base(self, expr: IndexRemappingBase,
                                  *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.array, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_index_base(self, expr: IndexBase, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.array, *args, **kwargs)

        self.rec_idx_or_size_tuple(expr.indices, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_basic_index(self, expr: BasicIndex, *args: Any, **kwargs: Any) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      *args: Any, **kwargs: Any) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          *args: Any, **kwargs: Any) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_stack(self, expr: Stack, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.arrays:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_concatenate(self, expr: Concatenate, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.arrays:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_einsum(self, expr: Einsum, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.args:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays,
                                 *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr._data.values():
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder,
            *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.send.data, *args, **kwargs)
        self.rec(expr.passthrough_data, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_named_array(self, expr: NamedArray, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr._container, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_loopy_call(self, expr: LoopyCall, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_function_definition(self, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        new_mapper = self.clone_for_callee(expr)
        for subexpr in expr.returns.values():
            new_mapper(subexpr, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_call(self, expr: Call, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.map_function_definition(expr.function, *args, **kwargs)
        for bnd in expr.bindings.values():
            self.rec(bnd, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr._container, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

# }}}


# {{{ CachedWalkMapper

class CachedWalkMapper(WalkMapper):
    """
    WalkMapper that visits each node in the DAG exactly once. This loses some
    information compared to :class:`WalkMapper` as a node is visited only from
    one of its predecessors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._visited_arrays_or_names: Set[Any] = set()
        self._visited_functions: Set[Any] = set()

    def get_cache_key(self, expr: ArrayOrNames, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_func_def_cache_key(
            self, expr: FunctionDefinition, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def rec(self, expr: ArrayOrNames, *args: Any, **kwargs: Any
            ) -> None:
        cache_key = self.get_cache_key(expr, *args, **kwargs)
        if cache_key in self._visited_arrays_or_names:
            return

        super().rec(expr, *args, **kwargs)
        self._visited_arrays_or_names.add(cache_key)

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        return type(self)()

    def map_function_definition(self, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> None:
        cache_key = self.get_func_def_cache_key(expr, *args, **kwargs)
        if (
                not self.visit(expr, *args, **kwargs)
                or cache_key in self._visited_functions):
            return

        new_mapper = self.clone_for_callee(expr)
        for subexpr in expr.returns.values():
            new_mapper(subexpr, *args, **kwargs)

        self._visited_functions.add(cache_key)

        self.post_visit(expr, *args, **kwargs)

# }}}


# {{{ TopoSortMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class TopoSortMapper(CachedWalkMapper):
    """A mapper that creates a list of nodes in topological order.

    :members: topological_order

    .. note::

        Does not consider the nodes inside  a
        :class:`~pytato.function.FunctionDefinition`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.topological_order: List[Array] = []

    def get_cache_key(self, expr: ArrayOrNames) -> int:
        return id(expr)

    def post_visit(self, expr: Any) -> None:
        self.topological_order.append(expr)

    def map_function_definition(self, expr: FunctionDefinition) -> None:
        # do nothing as it includes arrays from a different namespace.
        return

# }}}


# {{{ MapAndCopyMapper

class CachedMapAndCopyMapper(CopyMapper):
    """
    Mapper that applies *map_fn* to each node and copies it. Results of
    traversals are memoized i.e. each node is mapped via *map_fn* exactly once.
    """

    def __init__(self, map_fn: Callable[[ArrayOrNames], ArrayOrNames]) -> None:
        super().__init__()
        self.map_fn: Callable[[ArrayOrNames], ArrayOrNames] = map_fn

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__ and does not have map_fn
        return type(self)(self.map_fn)  # type: ignore[call-arg,attr-defined]

    # FIXME: It's awkward to have to duplicate all of this from TransformMapper;
    # figure out a better way
    def rec(self, expr: MappedT) -> MappedT:
        key = self.get_cache_key(expr)
        try:
            result = self._cache[key]
        except KeyError:
            pass
        else:
            if self._err_on_collision and expr is not self._seen_exprs[key]:
                raise ValueError(
                    f"cache collision detected on {type(expr)} in {type(self)}.")
            return result  # type: ignore[return-value]

        if self._err_on_collision:
            self._seen_exprs[key] = expr

        result = Mapper.rec(self, self.map_fn(expr))
        result_key = self.get_cache_key(result)

        # This only works if the expression has no existing duplicates (hence
        # the err_on_collision=True requirement). Otherwise, rec() could produce a
        # valid result that is not identical to expr due to deduplication
        if (
                self._err_on_duplication
                and hash(result_key) == hash(key)
                and result_key == key
                and result is not expr):
            raise ValueError(
                f"array duplication detected on {type(expr)} in {type(self)}."
                f"{id(result)=}, {id(expr)=}")

        try:
            result = self._seen_results[result_key]
        except KeyError:
            self._seen_results[result_key] = result

        self._cache[key] = result

        return result  # type: ignore[return-value]

    if TYPE_CHECKING:
        def __call__(self, expr: MappedT) -> MappedT:
            return self.rec(expr)

# }}}


# {{{ MPMS materializer

@dataclass(frozen=True, eq=True)
class MPMSMaterializerAccumulator:
    """This class serves as the return value of :class:`MPMSMaterializer`. It
    contains the set of materialized predecessors and the rewritten expression
    (i.e. the expression with tags for materialization applied).
    """
    materialized_predecessors: FrozenSet[Array]
    expr: Array


def _materialize_if_mpms(expr: Array,
                         nsuccessors: int,
                         predecessors: Iterable[MPMSMaterializerAccumulator]
                         ) -> MPMSMaterializerAccumulator:
    """
    Returns an instance of :class:`MPMSMaterializerAccumulator`, that
    materializes *expr* if it has more than 1 successor and more than 1
    materialized predecessor.
    """
    from functools import reduce

    materialized_predecessors: FrozenSet[Array] = reduce(
                                                    frozenset.union,
                                                    (pred.materialized_predecessors
                                                     for pred in predecessors),
                                                    frozenset())
    if nsuccessors > 1 and len(materialized_predecessors) > 1:
        new_expr = expr.tagged(ImplStored())
        return MPMSMaterializerAccumulator(frozenset([new_expr]), new_expr)
    else:
        return MPMSMaterializerAccumulator(materialized_predecessors, expr)


class MPMSMaterializer(Mapper):
    """
    See :func:`materialize_with_mpms` for an explanation.

    .. attribute:: nsuccessors

        A mapping from a node in the expression graph (i.e. an
        :class:`~pytato.Array`) to its number of successors.
    """
    def __init__(self, nsuccessors: Mapping[Array, int]):
        super().__init__()
        self.nsuccessors = nsuccessors
        self.cache: Dict[ArrayOrNames, MPMSMaterializerAccumulator] = {}

    # type-ignore reason: return type not compatible with Mapper.rec's type
    def rec(self, expr: ArrayOrNames) -> MPMSMaterializerAccumulator:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: MPMSMaterializerAccumulator = super().rec(expr)
        self.cache[expr] = result
        return result

    def _map_input_base(self, expr: InputArgumentBase
                        ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_named_array(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        raise NotImplementedError("only LoopyCallResult named array"
                                  " supported for now.")

    def map_index_lambda(self, expr: IndexLambda) -> MPMSMaterializerAccumulator:
        children_rec = {bnd_name: self.rec(bnd)
                        for bnd_name, bnd in sorted(expr.bindings.items())}

        new_expr = IndexLambda(expr=expr.expr,
                               shape=expr.shape,
                               dtype=expr.dtype,
                               bindings=immutabledict({bnd_name: bnd.expr
                                for bnd_name, bnd in sorted(children_rec.items())}),
                               axes=expr.axes,
                               var_to_reduction_descr=expr.var_to_reduction_descr,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    children_rec.values())

    def map_stack(self, expr: Stack) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_expr = Stack(tuple(ary.expr for ary in rec_arrays),
                         expr.axis, axes=expr.axes, tags=expr.tags,
                         non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_concatenate(self, expr: Concatenate) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_expr = Concatenate(tuple(ary.expr for ary in rec_arrays),
                               expr.axis,
                               axes=expr.axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_roll(self, expr: Roll) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = Roll(rec_array.expr, expr.shift, expr.axis, axes=expr.axes,
                        tags=expr.tags,
                        non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    (rec_array,))

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = AxisPermutation(rec_array.expr, expr.axis_permutation,
                                   axes=expr.axes, tags=expr.tags,
                                   non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,))

    def _map_index_base(self, expr: IndexBase) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        rec_indices = {i: self.rec(idx)
                       for i, idx in enumerate(expr.indices)
                       if isinstance(idx, Array)}

        new_expr = type(expr)(rec_array.expr,
                              tuple(rec_indices[i].expr
                                    if i in rec_indices
                                    else expr.indices[i]
                                    for i in range(
                                        len(expr.indices))),
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,) + tuple(rec_indices.values())
                                    )

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self, expr: Reshape) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = Reshape(rec_array.expr, expr.newshape,
                           expr.order, axes=expr.axes, tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,))

    def map_einsum(self, expr: Einsum) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.args]
        new_expr = Einsum(expr.access_descriptors,
                          tuple(ary.expr for ary in rec_arrays),
                          expr.redn_axis_to_redn_descr,
                          expr.index_to_access_descr,
                          axes=expr.axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays
                                 ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError

    def map_loopy_call_result(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        # loopy call result is always materialized
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> MPMSMaterializerAccumulator:
        rec_passthrough = self.rec(expr.passthrough_data)
        rec_send_data = self.rec(expr.send.data)
        new_expr = DistributedSendRefHolder(
            send=DistributedSend(rec_send_data.expr,
                                 dest_rank=expr.send.dest_rank,
                                 comm_tag=expr.send.comm_tag,
                                 tags=expr.send.tags),
            passthrough_data=rec_passthrough.expr,
            axes=rec_passthrough.expr.axes,
            tags=expr.tags,
            non_equality_tags=expr.non_equality_tags,
            )
        return MPMSMaterializerAccumulator(
            rec_passthrough.materialized_predecessors, new_expr)

    def map_distributed_recv(self, expr: DistributedRecv
                             ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_named_call_result(self, expr: NamedCallResult
                              ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError("MPMSMaterializer does not support functions.")

# }}}


# {{{ mapper frontends

def copy_dict_of_named_arrays(source_dict: DictOfNamedArrays,
        copy_mapper: CopyMapper) -> DictOfNamedArrays:
    """Copy the elements of a :class:`~pytato.DictOfNamedArrays` into a
    :class:`~pytato.DictOfNamedArrays`.

    :param source_dict: The :class:`~pytato.DictOfNamedArrays` to copy
    :param copy_mapper: A mapper that performs copies different array types
    :returns: A new :class:`~pytato.DictOfNamedArrays` containing copies of the
        items in *source_dict*
    """
    if not source_dict:
        data = {}
    else:
        data = {name: copy_mapper(val.expr)
                for name, val in sorted(source_dict.items())}

    return DictOfNamedArrays(data, tags=source_dict.tags)


def get_dependencies(expr: DictOfNamedArrays) -> Dict[str, FrozenSet[Array]]:
    """Returns the dependencies of each named array in *expr*.
    """
    dep_mapper = DependencyMapper()

    return {name: dep_mapper(val.expr) for name, val in expr.items()}


def map_and_copy(expr: MappedT,
                 map_fn: Callable[[ArrayOrNames], ArrayOrNames]
                 ) -> MappedT:
    """
    Returns a copy of *expr* with every array expression reachable from *expr*
    mapped via *map_fn*.

    .. note::

        Uses :class:`CachedMapAndCopyMapper` under the hood and because of its
        caching nature each node is mapped exactly once.
    """
    return CachedMapAndCopyMapper(map_fn)(expr)


def materialize_with_mpms(expr: DictOfNamedArrays) -> DictOfNamedArrays:
    r"""
    Materialize nodes in *expr* with MPMS materialization strategy.
    MPMS stands for Multiple-Predecessors, Multiple-Successors.

    .. note::

        - MPMS materialization strategy is a greedy materialization algorithm in
          which any node with more than 1 materialized predecessor and more than
          1 successor is materialized.
        - Materializing here corresponds to tagging a node with
          :class:`~pytato.tags.ImplStored`.
        - Does not attempt to materialize sub-expressions in
          :attr:`pytato.Array.shape`.

    .. warning::

        This is a greedy materialization algorithm and thereby this algorithm
        might be too eager to materialize. Consider the graph below:

        ::

                           I1          I2
                            \         /
                             \       /
                              \     /
                               🡦   🡧
                                 T
                                / \
                               /   \
                              /     \
                             🡧       🡦
                            O1        O2

        where, 'I1', 'I2' correspond to instances of
        :class:`pytato.array.InputArgumentBase`, and, 'O1' and 'O2' are the outputs
        required to be evaluated in the computation graph. MPMS materialization
        algorithm will materialize the intermediate node 'T' as it has 2
        predecessors and 2 successors. However, the total number of memory
        accesses after applying MPMS goes up as shown by the table below.

        ======  ========  =======
        ..        Before    After
        ======  ========  =======
        Reads          4        4
        Writes         2        3
        Total          6        7
        ======  ========  =======

    """
    from pytato.analysis import get_nusers
    materializer = MPMSMaterializer(get_nusers(expr))
    new_data = {}
    for name, ary in expr.items():
        new_data[name] = materializer(ary.expr).expr

    return DictOfNamedArrays(new_data, tags=expr.tags)

# }}}


# {{{ UsersCollector

class UsersCollector(CachedMapper[ArrayOrNames]):
    """
    Maps a graph to a dictionary representation mapping a node to its users,
    i.e. all the nodes using its value.

    .. attribute:: node_to_users

       Mapping of each node in the graph to its users.

    .. automethod:: __init__
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_to_users: Dict[ArrayOrNames,
                Set[Union[DistributedSend, ArrayOrNames]]] = {}

    # type-ignore-reason: incompatible with superclass (args/kwargs, return type)
    def __call__(self, expr: ArrayOrNames) -> None:  # type: ignore[override]
        # Root node has no predecessor
        self.node_to_users[expr] = set()
        self.rec(expr)

    def rec_idx_or_size_tuple(
            self, expr: Array, situp: Tuple[IndexOrShapeExpr, ...]
            ) -> None:
        for dim in situp:
            if isinstance(dim, Array):
                self.node_to_users.setdefault(dim, set()).add(expr)
                self.rec(dim)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        for child in expr._data.values():
            self.node_to_users.setdefault(child, set()).add(expr)
            self.rec(child)

    def map_named_array(self, expr: NamedArray) -> None:
        self.node_to_users.setdefault(expr._container, set()).add(expr)
        self.rec(expr._container)

    def map_einsum(self, expr: Einsum) -> None:
        for arg in expr.args:
            self.node_to_users.setdefault(arg, set()).add(expr)
            self.rec(arg)

        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_reshape(self, expr: Reshape) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_placeholder(self, expr: Placeholder) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_concatenate(self, expr: Concatenate) -> None:
        for ary in expr.arrays:
            self.node_to_users.setdefault(ary, set()).add(expr)
            self.rec(ary)

    def map_stack(self, expr: Stack) -> None:
        for ary in expr.arrays:
            self.node_to_users.setdefault(ary, set()).add(expr)
            self.rec(ary)

    def map_roll(self, expr: Roll) -> None:
        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_size_param(self, expr: SizeParam) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_axis_permutation(self, expr: AxisPermutation) -> None:
        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_data_wrapper(self, expr: DataWrapper) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_index_lambda(self, expr: IndexLambda) -> None:
        for child in expr.bindings.values():
            self.node_to_users.setdefault(child, set()).add(expr)
            self.rec(child)

        self.rec_idx_or_size_tuple(expr, expr.shape)

    def _map_index_base(self, expr: IndexBase) -> None:
        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

        for idx in expr.indices:
            if isinstance(idx, Array):
                self.node_to_users.setdefault(idx, set()).add(expr)
                self.rec(idx)

    def map_basic_index(self, expr: BasicIndex) -> None:
        self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> None:
        self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> None:
        self._map_index_base(expr)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.node_to_users.setdefault(child, set()).add(expr)
                self.rec(child)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> None:
        self.node_to_users.setdefault(expr.passthrough_data, set()).add(expr)
        self.rec(expr.passthrough_data)
        self.node_to_users.setdefault(expr.send.data, set()).add(expr.send)
        self.rec(expr.send.data)

    def map_distributed_recv(self, expr: DistributedRecv) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition, *args: Any
                                ) -> None:
        raise AssertionError("Control shouldn't reach at this point."
                             " Instantiate another UsersCollector to"
                             " traverse the callee function.")

    def map_call(self, expr: Call, *args: Any) -> None:
        for bnd in expr.bindings.values():
            self.rec(bnd)

    def map_named_call_result(self, expr: NamedCallResult, *args: Any) -> None:
        assert isinstance(expr._container, Call)
        for bnd in expr._container.bindings.values():
            self.node_to_users.setdefault(bnd, set()).add(expr)

        self.rec(expr._container)


def get_users(expr: ArrayOrNames) -> Dict[ArrayOrNames,
                                          Set[ArrayOrNames]]:
    """
    Returns a mapping from node in *expr* to its direct users.
    """
    user_collector = UsersCollector()
    user_collector(expr)
    return user_collector.node_to_users  # type: ignore[return-value]

# }}}


# {{{ operations on graphs in dict form

def _recursively_get_all_users(
        direct_users: Mapping[ArrayOrNames, Set[ArrayOrNames]],
        node: ArrayOrNames) -> FrozenSet[ArrayOrNames]:
    result = set()
    queue = list(direct_users.get(node, set()))
    ids_already_noted_to_visit: Set[int] = set()

    while queue:
        current_node = queue[0]
        queue = queue[1:]
        result.add(current_node)
        # visit each user only once.
        users_to_visit = frozenset({user
                                    for user in direct_users.get(current_node, set())
                                    if id(user) not in ids_already_noted_to_visit})

        ids_already_noted_to_visit.update({id(k)
                                           for k in users_to_visit})

        queue.extend(list(users_to_visit))

    return frozenset(result)


def rec_get_user_nodes(expr: ArrayOrNames,
                       node: ArrayOrNames,
                       ) -> FrozenSet[ArrayOrNames]:
    """
    Returns all direct and indirect users of *node* in *expr*.
    """
    users = get_users(expr)
    return _recursively_get_all_users(users, node)


def tag_user_nodes(
        graph: Mapping[ArrayOrNames, Set[ArrayOrNames]],
        tag: Any,
        starting_point: ArrayOrNames,
        node_to_tags: Optional[Dict[ArrayOrNames, Set[ArrayOrNames]]] = None
        ) -> Dict[ArrayOrNames, Set[Any]]:
    """Tags all nodes reachable from *starting_point* with *tag*.

    :param graph: A :class:`dict` representation of a directed graph, mapping each
        node to other nodes to which it is connected by edges. A possible
        use case for this function is the graph in
        :attr:`UsersCollector.node_to_users`.
    :param tag: The value to tag the nodes with.
    :param starting_point: A starting point in *graph*.
    :param node_to_tags: The resulting mapping of nodes to tags.
    """
    from warnings import warn
    warn("tag_user_nodes is set for deprecation in June, 2022",
         DeprecationWarning)

    if node_to_tags is None:
        node_to_tags = {}

    node_to_tags.setdefault(starting_point, set()).add(tag)

    for user in _recursively_get_all_users(graph, starting_point):
        node_to_tags.setdefault(user, set()).add(tag)

    return node_to_tags

# }}}


# {{{ deduplicate_data_wrappers

def _get_data_dedup_cache_key(ary: DataInterface) -> Hashable:
    import sys
    if "pyopencl" in sys.modules:
        from pyopencl.array import Array as CLArray
        from pyopencl import MemoryObjectHolder
        try:
            from pyopencl import SVMPointer
        except ImportError:
            SVMPointer = None  # noqa: N806

        if isinstance(ary, CLArray):
            base_data = ary.base_data
            if isinstance(ary.base_data, MemoryObjectHolder):
                ptr = base_data.int_ptr
            elif SVMPointer is not None and isinstance(base_data, SVMPointer):
                ptr = base_data.svm_ptr
            elif base_data is None:
                # pyopencl represents 0-long arrays' base_data as None
                ptr = None
            else:
                raise ValueError("base_data of array not understood")

            return (
                    ptr,
                    ary.offset,
                    ary.shape,
                    ary.strides,
                    ary.dtype,
                    )
    if isinstance(ary, np.ndarray):
        return (
                ary.__array_interface__["data"],
                ary.shape,
                ary.strides,
                ary.dtype,
                )
    else:
        raise NotImplementedError(str(type(ary)))


def deduplicate_data_wrappers(array_or_names: ArrayOrNames) -> ArrayOrNames:
    """For the expression graph given as *array_or_names*, replace all
    :class:`pytato.array.DataWrapper` instances containing identical data
    with a single instance.

    .. note::

        Currently only supports :class:`numpy.ndarray` and
        :class:`pyopencl.array.Array`.

    .. note::

        This function currently uses addresses of memory buffers to detect
        duplicate data, and so it may fail to deduplicate some instances
        of identical-but-separately-stored data. User code must tolerate
        this, but it must *also* tolerate this function doing a more thorough
        job of deduplication.
    """

    data_wrapper_cache: Dict[Hashable, DataWrapper] = {}
    data_wrappers_encountered = 0

    def cached_data_wrapper_if_present(ary: ArrayOrNames) -> ArrayOrNames:
        nonlocal data_wrappers_encountered

        if isinstance(ary, DataWrapper):
            data_wrappers_encountered += 1
            cache_key = _get_data_dedup_cache_key(ary.data)

            try:
                return data_wrapper_cache[cache_key]
            except KeyError:
                result = ary
                data_wrapper_cache[cache_key] = result
                return result
        else:
            return ary

    array_or_names = map_and_copy(array_or_names, cached_data_wrapper_if_present)

    # Remove any arrays that are now duplicates due to data wrapper deduplication
    array_or_names = CopyMapper(err_on_collision=False)(array_or_names)

    if data_wrappers_encountered:
        transform_logger.debug("data wrapper de-duplication: "
                               "%d encountered, %d kept, %d eliminated",
                               data_wrappers_encountered,
                               len(data_wrapper_cache),
                               data_wrappers_encountered - len(data_wrapper_cache))

    return array_or_names

# }}}


# vim: foldmethod=marker
