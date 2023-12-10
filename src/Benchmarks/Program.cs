// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Running;
using Benchmarks;

var tokenizeSpeed = BenchmarkRunner.Run<TokenizeSpeed>();
var sharpToken = BenchmarkRunner.Run<CompareToSharpToken>();
var otherLibs = BenchmarkRunner.Run<OtherLibs>();
