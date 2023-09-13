// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace FastBertTokenizer
{
    internal class UpperCollisionSolvingComparer : IEqualityComparer<string>
    {
        private readonly char[] LowerWithNonUniqueUpperCodepoint = new char[]
        {
            (char)73,
            (char)83,
            (char)105,
            (char)115,
            (char)181,
            (char)305,
            (char)383,
            (char)914,
            (char)917,
            (char)920,
            (char)921,
            (char)922,
            (char)924,
            (char)928,
            (char)929,
            (char)931,
            (char)934,
            (char)946,
            (char)949,
            (char)952,
            (char)953,
            (char)954,
            (char)956,
            (char)960,
            (char)961,
            (char)962,
            (char)963,
            (char)966,
            (char)976,
            (char)977,
            (char)981,
            (char)982,
            (char)1008,
            (char)1009,
            (char)1013,
            (char)1042,
            (char)1044,
            (char)1054,
            (char)1057,
            (char)1058,
            (char)1066,
            (char)1074,
            (char)1076,
            (char)1086,
            (char)1089,
            (char)1090,
            (char)1098,
            (char)1122,
            (char)1123,
            (char)7296,
            (char)7297,
            (char)7298,
            (char)7299,
            (char)7300,
            (char)7301,
            (char)7302,
            (char)7303,
            (char)7304,
            (char)7776,
            (char)7777,
            (char)7835,
            (char)8126,
            (char)42570,
            (char)42571,
        };

        public bool Equals(string x, string y)
        {
            if (x is not null && y is not null && (ContainsNonUnique(x.AsSpan()) || ContainsNonUnique(y.AsSpan())))
            {
                return StringComparer.Ordinal.Equals(x.ToLowerInvariant(), y.ToLowerInvariant());
            }

            return StringComparer.OrdinalIgnoreCase.Equals(x, y);
        }

        public int GetHashCode(string obj)
        {
            _ = obj ?? throw new ArgumentNullException(nameof(obj));

            if (ContainsNonUnique(obj.AsSpan()))
            {
                return StringComparer.Ordinal.GetHashCode(obj.ToLowerInvariant());
            }

            return StringComparer.OrdinalIgnoreCase.GetHashCode(obj);
        }

        private bool ContainsNonUnique(ReadOnlySpan<char> text)
        {
            foreach (var c in text)
            {
                if (Array.BinarySearch(LowerWithNonUniqueUpperCodepoint, c) >= 0)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
