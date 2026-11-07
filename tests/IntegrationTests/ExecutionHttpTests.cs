using System.Net;
using System.Text.Json;

namespace Aiursoft.CppRunner.Tests.IntegrationTests;

[TestClass]
public class ExecutionHttpTests : TestBase
{
    [TestMethod]
    [DataRow("cuda")]
    [DataRow("python-pytorch")]
    public async Task AnonymousGpuRequestsReturnUnauthorized(string language)
    {
        using var response = await Http.PostAsync("/runner/run?lang=" + language, new StringContent("not executed"));
        Assert.AreEqual(HttpStatusCode.Unauthorized, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<Models.CodeResult>();
        Assert.IsNotNull(result);
        Assert.AreEqual(401, result.ResultCode);
        StringAssert.Contains(result.Error, "Sign in");
    }

    [TestMethod]
    public async Task AnonymousMcpGpuCallIsAnErrorWithoutExecution()
    {
        using var response = await Mcp(new
        {
            jsonrpc = "2.0", id = 1, method = "tools/call",
            @params = new { name = "run_cuda", arguments = new { code = "not executed" } }
        });
        response.EnsureSuccessStatusCode();
        var text = await response.Content.ReadAsStringAsync();
        var json = text.StartsWith("data:", StringComparison.Ordinal) || text.StartsWith("event:", StringComparison.Ordinal)
            ? text.Split('\n').First(line => line.StartsWith("data: ", StringComparison.Ordinal))[6..]
            : text;
        using var document = JsonDocument.Parse(json);
        Assert.IsTrue(document.RootElement.GetProperty("result").GetProperty("isError").GetBoolean());
        StringAssert.Contains(text, "Sign in");
    }

    [TestMethod]
    public async Task McpAllowsFifteenRequestsThenReturns429WithoutLimitingHealth()
    {
        for (var i = 0; i < 16; i++)
        {
            using var response = await Mcp(new { jsonrpc = "2.0", id = i + 1, method = "tools/list" });
            Assert.AreEqual(i < 15 ? HttpStatusCode.OK : HttpStatusCode.TooManyRequests, response.StatusCode);
        }
        using var health = await Http.GetAsync("/health");
        health.EnsureSuccessStatusCode();
    }

    private Task<HttpResponseMessage> Mcp<T>(T body)
    {
        Http.DefaultRequestHeaders.Accept.Clear();
        Http.DefaultRequestHeaders.Accept.ParseAdd("application/json, text/event-stream");
        Http.DefaultRequestHeaders.Remove("MCP-Protocol-Version");
        Http.DefaultRequestHeaders.TryAddWithoutValidation("MCP-Protocol-Version", "2025-03-26");
        return Http.PostAsJsonAsync("/mcp", body);
    }
}
