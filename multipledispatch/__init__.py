import inspect
from functools import total_ordering
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union, get_args, get_origin, get_type_hints

@total_ordering
class Signature(tuple):
    def supercedes(self, other: "Signature"):
        """
        This signature is strictly more specific than 'other'.
        """
        return all(map(issubclass, self, other))

    def __lt__(self, other: "Signature"):
        if len(self) != len(other):
            return False

        return self.supercedes(other) and not other.supercedes(self)

    def __eq__(self, other: "Signature"):
        if len(self) != len(other):
            return False
        
        return all(a == b for a, b in zip(self, other))

    def __hash__(self):
        return super().__hash__()

def unpack_signatures(tp: Any, *args) -> Iterator[Signature]:
    """
    Expands a signature like (int, Union[int, str]) into signatures 
    (int, int) and (int, str).
    """
    GenericAlias = type(Iterable)

    if get_origin(tp) is Union:
        types = get_args(tp)
    elif isinstance(tp, GenericAlias) or not isinstance(tp, type):
        raise TypeError(f"Non-type annotation: {repr(tp)}")
    else:
        types = [tp]

    for tp in types:
        if args:
            for sig in unpack_signatures(*args):
                yield Signature(tuple([tp, *sig]))
        else:
            yield Signature(tuple([tp]))

def get_signatures(fn: Callable) -> List[Signature]:
    """
    Gets the types of the args of fn, unpacking Unions.
    """

    # inspect.signature() and get_type_hints() do two
    # different but complementary things.
    # inspect.signature() gets all parameters, even those
    # without type hints.
    # get_type_hints() automatically evaluates string
    # type hints, but skips parameters without type hints.
    # Ergo, we use both.
    params = inspect.signature(fn).parameters
    type_hints = get_type_hints(fn)
    
    param_types = []
    for param in params:
        param_type = type_hints.get(param, object)
        if param_type is Any:
            param_type = object
        param_types.append(param_type)

    return list(unpack_signatures(*param_types))

class multipledispatch:
    """
    Turns a function into a Multiple Dispatch Function.

    Ex:
    @multipledispatch
    def foo(a: object, b: object):
      print("Got two objects")

    @foo.register
    def foo(a: str, b: int):
        print("Got an string and an int")

    @foo.register
    def foo(a: str, b: str):
        print("Got a string and a string")

    @foo.register
    def foo(a: int, b: Union[int, str]):
        print("Got an int, and either an int or a string")
    """

    # Invariant: signature_map keys are in ascending order
    signature_map: Dict[Signature, Callable] = dict()

    def __init__(self, fn: Callable):
        self.register(fn)

    def __call__(self, *args):
        for signature, fn in self.signature_map.items():
            if len(signature) != len(args):
                continue
            if all(map(isinstance, args, signature)):
                return fn(*args)

        raise TypeError(f"No implementations found for args: {repr(args)}")

    def register(self, fn: Callable) -> Callable:
        """
        Registers a new implementation of this callable. Generally used as a decorator.
        """
        signatures = get_signatures(fn)

        for signature in signatures:
            if registered := self.signature_map.get(signature):
                raise TypeError(f"Multiple definitions of function for same parameter types: {registered} and {fn}")

        self._add_items([(signature, fn) for signature in signatures])

        return self

    def _add_items(self, new_items: List[Tuple[Signature, Callable]]):
        """
        Adds items to self.signature_map, preserving order.
        """

        # Insert each signature where it belongs in order
        # This will tend to be faster than just calling sort(),
        # unless there are a LOT of signatures to be inserted.
        # Additionally, Signature.__lt__ isn't a total ordering,
        # so there's no guarantee that sort() would work anyway!
        items = list(self.signature_map.items())
        for (signature, fn) in new_items:
            for i, item in enumerate(items):
                if signature < item[0]:
                    items.insert(i, (signature, fn))
                    break
            else:
                items.append((signature, fn))

        self.signature_map = dict(items)
