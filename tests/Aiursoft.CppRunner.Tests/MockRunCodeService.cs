using Aiursoft.Canon;
using Aiursoft.CppRunner.Models;
using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Services;
using Microsoft.Extensions.Logging;

namespace Aiursoft.CppRunner.Tests;

public class MockRunCodeService : RunCodeService
{
    public MockRunCodeService(ILogger<RunCodeService> logger, CommandService commandService, CanonQueue queue) : base(logger, commandService, queue)
    {
    }
    
    public override Task<CodeResult> RunCode(string code, ILang lang)
    {
        return Task.FromResult(new CodeResult
        {
            ResultCode = 0,
            Output = "Hello world!",
            Error = string.Empty
        });
    }
}