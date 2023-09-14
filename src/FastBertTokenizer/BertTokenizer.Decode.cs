// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text;

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    private Dictionary<long, string>? _decodePrefixes;
    private Dictionary<long, string>? _decodeSuffixes;

    public string Decode(ReadOnlySpan<long> tokenIds)
    {
        _ = _prefixes ?? throw new InvalidOperationException("Vocabulary not loaded.");
        _ = _suffixes ?? throw new InvalidOperationException("Vocabulary not loaded.");

        _decodeSuffixes ??= _suffixes.ToDictionary(x => x.Value, x => x.Key);
        if (_decodePrefixes is null)
        {
            _decodePrefixes = _prefixes.ToDictionary(x => x.Value, x => x.Key);
            _decodePrefixes.Add(_unk.Id, _unk.Token);
            _decodePrefixes.Add(_cls.Id, _cls.Token);
            _decodePrefixes.Add(_sep.Id, _sep.Token);
            _decodePrefixes.Add(_pad.Id, _pad.Token);
        }

        var sb = new StringBuilder();
        sb.Append(_decodePrefixes[tokenIds[0]]);
        foreach (var id in tokenIds.Slice(1))
        {
            if (_decodePrefixes.TryGetValue(id, out var prefix))
            {
                sb.Append(' ');
                sb.Append(prefix);
            }

            if (_decodeSuffixes.TryGetValue(id, out var suffix))
            {
                sb.Append(suffix);
            }
        }

        return sb.ToString();
    }
}
