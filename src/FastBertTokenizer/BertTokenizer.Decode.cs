// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text;

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    private Dictionary<long, string>? _decodePrefixes;
    private Dictionary<long, string>? _decodeSuffixes;

    public string Decode(ReadOnlySpan<long> tokenIds, bool cleanupTokenizationSpaces = true)
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

        if (tokenIds.Length == 0)
        {
            return string.Empty;
        }

        var sb = new StringBuilder();
        if (_decodePrefixes.TryGetValue(tokenIds[0], out var firstPrefix))
        {
            sb.Append(firstPrefix);
        }
        else
        {
            // Our decoded text does not start with a word start but in the middle of a word.
            sb.Append(_decodeSuffixes[tokenIds[0]]);
        }

        foreach (var id in tokenIds.Slice(1))
        {
            if (_decodePrefixes.TryGetValue(id, out var prefix))
            {
                if (!cleanupTokenizationSpaces || !EmitNoSpaceBefore(prefix))
                {
                    sb.Append(' ');
                }

                sb.Append(prefix);
            }

            if (_decodeSuffixes.TryGetValue(id, out var suffix))
            {
                sb.Append(suffix);
            }
        }

        // There is probably a faster implementation of this.
        // Decode isn't currently the focus though.
        if (cleanupTokenizationSpaces)
        {
            sb.Replace(" ' ", "'");
            sb.Replace(" n't", "n't");
            sb.Replace(" 'm", "'m");
            sb.Replace(" do not", "don't"); // while this seems strange this is what Hugging Face does
            sb.Replace(" 's", "'s");
            sb.Replace(" 've", "'ve");
            sb.Replace(" 're", "'re");
        }

        return sb.ToString();
    }

    // See https://github.com/huggingface/tokenizers/blob/daf361676bdfd14088f7e0bc087effc6a9cfdf3e/tokenizers/src/decoders/wordpiece.rs#L31
    private bool EmitNoSpaceBefore(string prefix)
    {
        return ".".Equals(prefix, StringComparison.Ordinal)
            || "?".Equals(prefix, StringComparison.Ordinal)
            || "!".Equals(prefix, StringComparison.Ordinal)
            || ",".Equals(prefix, StringComparison.Ordinal);
    }
}
