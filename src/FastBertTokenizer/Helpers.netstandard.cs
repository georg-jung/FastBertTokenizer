// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

#if NETSTANDARD
internal static partial class Helpers
{
    // taken from https://github.com/dotnet/machinelearning/blob/72cfdf611a510ba0570170a708ddcc1a1928f329/src/Microsoft.ML.Tokenizers/Utils/Helpers.netstandard.cs#L55
    internal static int GetHashCode(ReadOnlySpan<char> span)
    {
        var hash = 17;
        foreach (var c in span)
        {
            hash = (hash * 31) + c;
        }

        return hash;
    }
}
#endif
