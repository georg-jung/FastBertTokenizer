// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

#if NETSTANDARD2_0
internal static class Backports
{
    public static void Deconstruct(this KeyValuePair<string, int> kvp, out string key, out int value)
    {
        key = kvp.Key;
        value = kvp.Value;
    }
}
#endif
