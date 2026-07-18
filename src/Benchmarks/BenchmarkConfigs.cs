// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;

namespace Benchmarks;

internal static class BenchmarkDefaults
{
    /// <summary>
    /// The released FastBertTokenizer version that local changes are compared against.
    /// Keep in sync with the FastBertTokenizer PackageReference in Benchmarks.csproj.
    /// </summary>
    public const string BaselineNuGetVersion = "1.0.28";

    /// <summary>
    /// Set to "1" to run quick smoke jobs instead of the full, slow benchmark jobs.
    /// Program.cs sets this when the --smoke command line flag is passed.
    /// </summary>
    public const string SmokeEnvVar = "FASTBERTTOKENIZER_BENCH_SMOKE";

    public static bool IsSmoke => Environment.GetEnvironmentVariable(SmokeEnvVar) == "1";

    public static Job BaseJob => IsSmoke ? Job.ShortRun : Job.Default;

    public static Job LocalBuildJob => BaseJob.WithCustomBuildConfiguration("LocalBuild");
}

/// <summary>
/// Jobs for benchmarks that measure FastBertTokenizer itself: the local build on all
/// supported (non-EOL) runtimes and, in full mode, the released NuGet baseline version.
/// </summary>
internal sealed class SpeedConfig : ManualConfig
{
    public SpeedConfig()
    {
        AddJob(BenchmarkDefaults.LocalBuildJob.WithRuntime(CoreRuntime.Core10_0).WithId("local-net10.0"));
        AddJob(BenchmarkDefaults.LocalBuildJob.WithRuntime(CoreRuntime.Core80).WithId("local-net8.0"));

        if (!BenchmarkDefaults.IsSmoke)
        {
            // In the default (non-LocalBuild) build configuration, Benchmarks.csproj references
            // the released FastBertTokenizer baseline version from NuGet instead of the local project.
            var nugetJob = BenchmarkDefaults.BaseJob;
            AddJob(nugetJob.WithRuntime(CoreRuntime.Core10_0).WithId($"nuget-{BenchmarkDefaults.BaselineNuGetVersion}-net10.0"));
            AddJob(nugetJob.WithRuntime(CoreRuntime.Core80).WithId($"nuget-{BenchmarkDefaults.BaselineNuGetVersion}-net8.0"));
        }
    }
}

/// <summary>
/// Jobs for benchmarks that compare FastBertTokenizer to other libraries: just the local
/// build on the latest runtime, as the compared libraries manage their runtime behavior
/// themselves and we don't need a per-runtime breakdown for every competitor.
/// </summary>
internal sealed class CompareConfig : ManualConfig
{
    public CompareConfig()
    {
        AddJob(BenchmarkDefaults.LocalBuildJob.WithRuntime(CoreRuntime.Core10_0).WithId("local-net10.0"));
    }
}
