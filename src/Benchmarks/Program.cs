// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Running;
using Benchmarks;

// Usage:
//   dotnet run -c Release -f net10.0 -- --filter '*TokenizeSpeed*'           full benchmark of FastBertTokenizer itself
//   dotnet run -c Release -f net10.0 -- --filter '*OtherLibs*'               compare against other tokenizer libraries
//   dotnet run -c Release -f net10.0 -- --smoke --filter '*'                 quick smoke run (e.g. for CI on PRs)
//   dotnet run -c Release -f net10.0 -- --list flat                          list all available benchmarks
//
// --smoke is our own flag (it selects short-running jobs); everything else is passed to BenchmarkDotNet.
// See https://benchmarkdotnet.org/articles/guides/console-args.html for all supported arguments.
if (args.Contains("--smoke"))
{
    Environment.SetEnvironmentVariable(BenchmarkDefaults.SmokeEnvVar, "1");
    args = args.Where(a => a != "--smoke").ToArray();
}

var summaries = BenchmarkSwitcher.FromAssembly(typeof(TokenizeSpeed).Assembly).Run(args).ToList();

// BenchmarkDotNet reports failed benchmarks (e.g. generated project build failures or crashed
// benchmark processes) just as NA rows. Exit non-zero so CI notices; informational invocations
// like --list produce no summaries and still exit 0.
return summaries.Any(s => s.HasCriticalValidationErrors || s.Reports.Any(r => !r.Success)) ? 1 : 0;
