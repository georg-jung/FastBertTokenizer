# adapted from https://github.com/open-telemetry/opentelemetry-dotnet/blob/f2c225519db42e97c978bbb87b765da3204de860/build/test-aot-compatibility.ps1

param([string]$targetNetFramework)

$rootDirectory = $PSScriptRoot
$publishOutput = dotnet publish $rootDirectory/src/FastBertTokenizer.AotCompatibility.TestApp/FastBertTokenizer.AotCompatibility.TestApp.csproj -nodeReuse:false /p:UseSharedCompilation=false /p:ExposeExperimentalFeatures=true

$actualWarningCount = 0

foreach ($line in $($publishOutput -split "`r`n"))
{
    if ($line -like "*analysis warning IL*")
    {
        Write-Host $line

        $actualWarningCount += 1
    }
}

pushd $rootDirectory/bin/FastBertTokenizer.AotCompatibility.TestApp/Release/$targetNetFramework/linux-x64/publish

Write-Host "Executing test App..."
./FastBertTokenizer.AotCompatibility.TestApp
Write-Host "Finished executing test App"

if ($LastExitCode -ne 0)
{
    Write-Host "There was an error while executing AotCompatibility Test App. LastExitCode is:", $LastExitCode
}

popd

Write-Host "Actual warning count is:", $actualWarningCount
$expectedWarningCount = 0

$testPassed = 0
if ($actualWarningCount -ne $expectedWarningCount)
{
    $testPassed = 1
    Write-Host "Actual warning count:", actualWarningCount, "is not as expected. Expected warning count is:", $expectedWarningCount
}

Write-Host "Publish output:", $publishOutput

Exit $testPassed
