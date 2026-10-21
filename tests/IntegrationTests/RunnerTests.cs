using Aiursoft.Canon;
using System.Net;
using System.Runtime.InteropServices;
using Aiursoft.CSTools.Tools;

namespace Aiursoft.CppRunner.Tests.IntegrationTests;

/// <summary>
/// Tests for the language listing and default-code endpoints.
/// Migrated from Old/cpprunner/tests.
/// </summary>
[TestClass]
public class LangsControllerTests : TestBase
{
    [TestMethod]
    [DataRow("/langs")]
    [DataRow("/langs/bash/default")]
    [DataRow("/langs/c/default")]
    [DataRow("/langs/cpp/default")]
    [DataRow("/langs/csharp/default")]
    [DataRow("/langs/go/default")]
    [DataRow("/langs/haskell/default")]
    [DataRow("/langs/java/default")]
    [DataRow("/langs/lisp/default")]
    [DataRow("/langs/lua/default")]
    [DataRow("/langs/javascript/default")]
    [DataRow("/langs/perl/default")]
    [DataRow("/langs/python/default")]
    [DataRow("/langs/php/default")]
    [DataRow("/langs/powershell/default")]
    [DataRow("/langs/ruby/default")]
    [DataRow("/langs/rust/default")]
    [DataRow("/langs/swift/default")]
    [DataRow("/langs/typescript/default")]
    public async Task GetLang_ReturnsOk(string url)
    {
        var response = await Http.GetAsync(url);
        response.EnsureSuccessStatusCode();
    }

    [TestMethod]
    [DataRow("/langs/cpp22")]
    [DataRow("/langs/csss/default")]
    public async Task GetLang_UnknownLang_ReturnsNotFound(string url)
    {
        var response = await Http.GetAsync(url);
        Assert.AreEqual(HttpStatusCode.NotFound, response.StatusCode);
    }
}

/// <summary>
/// Tests for the /Runner/run endpoint.
/// Actual Docker execution is skipped on Windows and inside Docker containers.
/// Migrated from Old/cpprunner/tests.
/// </summary>
[TestClass]
public class RunnerControllerTests : TestBase
{
    [TestMethod]
    [DataRow("bash")]
    [DataRow("c")]
    [DataRow("cpp")]
    [DataRow("csharp")]
    [DataRow("go")]
    [DataRow("haskell")]
    [DataRow("java")]
    [DataRow("lisp")]
    [DataRow("lua")]
    [DataRow("javascript")]
    [DataRow("perl")]
    [DataRow("python")]
    [DataRow("php")]
    [DataRow("powershell")]
    [DataRow("ruby")]
    [DataRow("rust")]
    [DataRow("swift")]
    [DataRow("typescript")]
    public async Task RunCode_DefaultCode_ReturnsOkWithFibonacciOutput(string lang)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Assert.Inconclusive("Docker-based tests are not supported on Windows.");
            return;
        }

        if (EntryExtends.IsInDocker())
        {
            Assert.Inconclusive("Docker-based tests are not supported inside Docker.");
            return;
        }

        if (Server != null)
        {
            await Server.PullContainersAsync();
        }

        var defaultCode = await Http.GetStringAsync($"/langs/{lang}/default");
        var response = await Http.PostAsync(
            $"/runner/run?lang={lang}",
            new StringContent(defaultCode));

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<Models.CodeResult>();
        Assert.IsNotNull(result);
        Assert.AreEqual(0, result.ResultCode);
        StringAssert.Contains(result.Output, "55");
    }

    [TestMethod]
    public async Task MissingRuntimeReturnsClearTemporaryFailureWithoutExecutingCode()
    {
        var images = new RuntimeImageWarmupTests.ImageClient();
        var services = Server!.Services;
        var runner = new Services.RunCodeService(
            services.GetRequiredService<IConfiguration>(),
            services.GetRequiredService<Services.HasGpuService>(),
            services.GetRequiredService<ILogger<Services.RunCodeService>>(),
            services.GetRequiredService<Aiursoft.CSTools.Services.CommandService>(),
            services.GetRequiredService<CanonQueue>(), images,
            services.GetRequiredService<IHostApplicationLifetime>(),
            services.GetRequiredService<Microsoft.Extensions.Localization.IStringLocalizer<Services.RunCodeService>>());
        var result = await runner.RunCode("This code must not execute.", new Lang.CLang());
        Assert.AreEqual(75, result.ResultCode);
        StringAssert.Contains(result.Error, "not ready");
        Assert.AreEqual(result.Error, result.Output);
        Assert.AreEqual(1, images.Inspections);
        Assert.IsEmpty(images.Pulled);
    }

    [TestMethod]
    public async Task RunCode_UnknownLang_ReturnsNotFound()
    {
        var response = await Http.PostAsync(
            "/runner/run?lang=cpp22",
            new StringContent("int main() { return 0; }"));

        Assert.AreEqual(HttpStatusCode.NotFound, response.StatusCode);
    }
}
