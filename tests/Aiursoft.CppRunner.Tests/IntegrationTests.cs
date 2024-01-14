using System.Net;
using System.Runtime.InteropServices;
using Aiursoft.CSTools.Tools;
using Aiursoft.WebTools.Attributes;
using Microsoft.Extensions.Hosting;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Aiursoft.WebTools.Extends;

namespace Aiursoft.CppRunner.Tests;

[TestClass]
public class IntegrationTests
{
    private readonly string _endpointUrl;
    private readonly int _port;
    private readonly HttpClient _http;
    private IHost? _server;

    public IntegrationTests()
    {
        LimitPerMin.GlobalEnabled = false;
        _port = Network.GetAvailablePort();
        _endpointUrl = $"http://localhost:{_port}";
        _http = new HttpClient();
    }

    [TestInitialize]
    public async Task CreateServer()
    {
        _server = App<Startup>(Array.Empty<string>(), port: _port);
        await _server.StartAsync();
    }

    [TestCleanup]
    public async Task CleanServer()
    {
        if (_server == null) return;
        await _server.StopAsync();
        _server.Dispose();
    }

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
    [DataRow("/langs/python/default")]
    [DataRow("/langs/ruby/default")]
    [DataRow("/langs/rust/default")]
    [DataRow("/langs/swift/default")]
    [DataRow("/langs/typescript/default")]
    public async Task GetLang(string url)
    {
        var response = await _http.GetAsync(_endpointUrl + url);
        response.EnsureSuccessStatusCode(); // Status Code 200-299
    }
    
    [TestMethod]
    [DataRow("/langs/cpp22")]
    [DataRow("/langs/csss/default")]
    public async Task GetLangNotFound(string url)
    {
        var response = await _http.GetAsync(_endpointUrl + url);
        Assert.AreEqual(HttpStatusCode.NotFound, response.StatusCode);
    }

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
    [DataRow("python")]
    [DataRow("ruby")]
    [DataRow("rust")]
    [DataRow("swift")]
    [DataRow("typescript")]
    public async Task RunCode(string lang)
    {
        // Skip this test on Windows.
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Assert.Inconclusive("This test is not supported on Windows.");
        }
        
        var defaultCode = await _http.GetStringAsync(_endpointUrl + $"/langs/{lang}/default");
        var response = await _http.PostAsync(_endpointUrl + $"/runner/run?lang={lang}", new StringContent(defaultCode));
        response.EnsureSuccessStatusCode(); // Status Code 200-299

        var message = await response.Content.ReadAsStringAsync();
        Assert.IsTrue(message.Contains("1"));
        Assert.IsTrue(message.Contains("2"));
        Assert.IsTrue(message.Contains("3"));
        Assert.IsTrue(message.Contains("5"));
        Assert.IsTrue(message.Contains("8"));
        Assert.IsTrue(message.Contains("13"));
        Assert.IsTrue(message.Contains("21"));
        Assert.IsTrue(message.Contains("34"));
        Assert.IsTrue(message.Contains("55"));
    }
    
    [TestMethod]
    public async Task RunCodeNotFound()
    {
        var response = await _http.PostAsync(_endpointUrl + "/runner/run?lang=cpp22", new StringContent("int main() { return 0; }"));
        Assert.AreEqual(HttpStatusCode.NotFound, response.StatusCode);
    }
}
