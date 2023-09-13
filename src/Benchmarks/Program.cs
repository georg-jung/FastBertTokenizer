// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using Benchmarks;

var summary = BenchmarkRunner.Run<TokenizeSpeed>();
