using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Tests;

public class MockCommandService : CommandService
{
    public override Task<(int code, string output, string error)> RunCommandAsync(string bin, string arg, string path, TimeSpan? timeout = null, bool killTimeoutProcess = true)
    {
        return Task.FromResult((0, "Hello world!", string.Empty));
    }
}