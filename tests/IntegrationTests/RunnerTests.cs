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

        var message = await response.Content.ReadAsStringAsync();
        // All default codes produce Fibonacci numbers; verify a few are present
        Assert.IsTrue(message.Contains("1"),  $"[{lang}] Missing '1' in output: {message}");
        Assert.IsTrue(message.Contains("2"),  $"[{lang}] Missing '2' in output: {message}");
        Assert.IsTrue(message.Contains("3"),  $"[{lang}] Missing '3' in output: {message}");
        Assert.IsTrue(message.Contains("5"),  $"[{lang}] Missing '5' in output: {message}");
        Assert.IsTrue(message.Contains("8"),  $"[{lang}] Missing '8' in output: {message}");
        Assert.IsTrue(message.Contains("13"), $"[{lang}] Missing '13' in output: {message}");
        Assert.IsTrue(message.Contains("21"), $"[{lang}] Missing '21' in output: {message}");
        Assert.IsTrue(message.Contains("34"), $"[{lang}] Missing '34' in output: {message}");
        Assert.IsTrue(message.Contains("55"), $"[{lang}] Missing '55' in output: {message}");
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
