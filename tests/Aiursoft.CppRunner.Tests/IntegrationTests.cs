using Aiursoft.CSTools.Tools;
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
        _port = Network.GetAvailablePort();
        _endpointUrl = $"http://localhost:{_port}";
        _http = new HttpClient();
    }

    [TestInitialize]
    public async Task CreateServer()
    {
        _server = App<TestStartup>(Array.Empty<string>(), port: _port);
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
    [DataRow("/langs/cpp/default")]
    [DataRow("/langs/csharp/default")]
    public async Task GetHome(string url)
    {
        var response = await _http.GetAsync(_endpointUrl + url);
        response.EnsureSuccessStatusCode(); // Status Code 200-299
    }

    [TestMethod]
    [DataRow("cpp", "int main() { return 0; }")]
    [DataRow("csharp", "public class Program { public static void Main() { } }")]
    public async Task RunCode(string lang, string code)
    {
        var response = await _http.PostAsync(_endpointUrl + $"/runner/run?lang={lang}", new StringContent(code));
        response.EnsureSuccessStatusCode(); // Status Code 200-299
    }
}
