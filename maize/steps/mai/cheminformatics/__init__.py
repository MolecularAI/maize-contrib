"""
Various cheminformatics utilities

"""

try:
    from maize.utilities.utilities import deprecated
except ImportError:
    from collections.abc import Callable
    from typing import Any

    def deprecated(_: str) -> Callable[[Any], Any]:  # type: ignore
        def inner(obj: Any) -> Any:
            return obj

        return inner


from .filters import (
    BestIsomerFilter,
    BestConformerFilter,
    RankingFilter,
    TagFilter,
    SMARTSFilter,
    ChargeFilter,
)
from .sorters import TagSorter
from .taggers import (
    TagAgg,
    TagMath,
    SetTag,
    SetName,
    TagIndex,
    SortByTag,
    SetScoreTag,
    ExtractScores,
    RMSD,
    LogTags,
    ExtractTag,
)


@deprecated("please use 'TagFilter' instead")
class IsomerCollectionTagFilter(TagFilter):
    pass


@deprecated("please use 'RankingFilter' instead")
class IsomerCollectionRankingFilter(RankingFilter):
    pass


@deprecated("please use 'SMARTSFilter' instead")
class IsomerFilter(SMARTSFilter):
    pass


__all__ = [
    "BestIsomerFilter",
    "BestConformerFilter",
    "IsomerCollectionTagFilter",
    "IsomerCollectionRankingFilter",
    "IsomerFilter",
    "TagFilter",
    "RankingFilter",
    "SMARTSFilter",
    "ChargeFilter",
    "TagSorter",
    "TagAgg",
    "TagMath",
    "SetTag",
    "SetName",
    "TagIndex",
    "LogTags",
    "SortByTag",
    "SetScoreTag",
    "ExtractScores",
    "ExtractTag",
    "RMSD",
]
