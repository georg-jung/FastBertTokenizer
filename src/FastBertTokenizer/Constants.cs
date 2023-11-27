// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer
{
    internal static class Constants
    {
        // https://github.com/dotnet/runtime/blob/6ed57ec3d26353549d7bffe5055a7100f443ff5b/src/libraries/System.Text.Json/Common/JsonConstants.cs#L12C33-L12C33
        public const int StackallocByteThreshold = 256;
        public const int StackallocCharThreshold = StackallocByteThreshold / 2;
    }
}
