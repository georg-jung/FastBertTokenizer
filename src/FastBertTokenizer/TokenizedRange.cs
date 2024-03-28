// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace FastBertTokenizer;

/// <summary>
/// Represents information about a range of an input that was tokenized.
/// </summary>
/// <typeparam name="TKey">A user-defined key type that identifies the tokenized input.</typeparam>
/// <param name="Key">The key that identifies the input that was tokenized.</param>
/// <param name="Offset">The corresponding tokenization result's first token represents a word at this offset in the input.</param>
/// <param name="LastTokenizedWordStartIndex">
/// Null, if the remaining input was fully tokenized. The index of the first char of the word that was cut off otherwise.
/// This could be the offset value for the next tokenized range.
/// </param>
[Experimental("FBERTTOK001")]
public record struct TokenizedRange<TKey>(TKey Key, int Offset, int? LastTokenizedWordStartIndex)
{
}
