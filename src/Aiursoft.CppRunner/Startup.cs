using System.Reflection;
using System.Text.Json;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Services;
using Aiursoft.WebTools.Abstractions.Models;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;

namespace Aiursoft.CppRunner;

public class Startup : IWebStartup
{
    public virtual void ConfigureServices(IConfiguration configuration, IWebHostEnvironment environment,
        IServiceCollection services)
    {
        services.AddScoped<CommandService>();
        services.AddScoped<RunCodeService>();

        services.AddTaskCanon();
        services.AddScoped<ILang, CLang>();
        services.AddScoped<ILang, CppLang>();
        services.AddScoped<ILang, CudaLang>();
        services.AddScoped<ILang, CSharpLang>();

        services.AddScoped<ILang, GoLang>();
        services.AddScoped<ILang, RustLang>();

        services.AddScoped<ILang, JavaScriptLang>();
        services.AddScoped<ILang, TypeScriptLang>();

        services.AddScoped<ILang, PythonLang>();
        services.AddScoped<ILang, PythonWithPytorch>();
        services.AddScoped<ILang, BashLang>();
        services.AddScoped<ILang, PowerShellLang>();

        services.AddScoped<ILang, SwiftLang>();

        services.AddScoped<ILang, JavaLang>();
        services.AddScoped<ILang, RubyLang>();

        services.AddScoped<ILang, PhpLang>();

        services.AddScoped<ILang, PerlLang>();
        services.AddScoped<ILang, LuaLang>();

        services.AddScoped<ILang, HaskellLang>();
        services.AddScoped<ILang, LispLang>();


        services
            .AddControllersWithViews()
            .AddApplicationPart(Assembly.GetExecutingAssembly());

        services
            .AddMcpServer(options =>
            {
                options.ServerInfo = new Implementation
                {
                    Name = "Code Runner",
                    Version = "1.0.0"
                };
                options.Capabilities = new ServerCapabilities
                {
                    Tools = new ToolsCapability
                    {
                        ListToolsHandler = (request, _) =>
                        {
                            var langs = request.Services!.GetRequiredService<IEnumerable<ILang>>();
                            var logger = request.Services!.GetRequiredService<ILogger<Startup>>();
                            logger.LogInformation("List tools...");
                            var tools = langs.Select(l => new Tool
                                {
                                    Name = $"run_{l.LangName}",
                                    Description = $"Compile and run {l.LangDisplayName} code and get output and error results.",
                                    InputSchema = JsonSerializer.Deserialize<JsonElement>(@"
                                    {
                                      ""type"":""object"",
                                      ""properties"": {
                                        ""code"": {
                                          ""type"": ""string"",
                                          ""description"": ""The source code to execute""
                                        }
                                     },
                                     ""required"": [""code""]
                               }")
                                })
                                .ToList();
                            return ValueTask.FromResult(new ListToolsResult { Tools = tools });
                        },
                        CallToolHandler = async (request, _) =>
                        {
                            var runCodeService = request.Services!.GetRequiredService<RunCodeService>();
                            var logger = request.Services!.GetRequiredService<ILogger<RunCodeService>>();
                            var langs = request.Services!.GetRequiredService<IEnumerable<ILang>>();
                            var toolName = request.Params?.Name
                                           ?? throw new McpException("Missing tool name");

                            logger.LogInformation("Call tool {ToolName}...", toolName);
                            var langKey = toolName["run_".Length..];
                            var langImpl = langs.FirstOrDefault(l =>l.LangName.Equals(langKey, StringComparison.OrdinalIgnoreCase))
                                           ?? throw new McpException($"Unknown language '{langKey}'");

                            var code = request.Params.Arguments?["code"].ToString()
                                       ?? throw new McpException("Missing argument 'code'");

                            try
                            {
                                logger.LogInformation("Running code in {LangName}...", langImpl.LangDisplayName);
                                var result = await runCodeService.RunCode(code, langImpl);
                                logger.LogInformation("Code run completed with result code {ResultCode}.", result.ResultCode);
                                return new CallToolResult
                                {
                                    Content =
                                    [
                                        new TextContentBlock
                                        {
                                            Text = $"Result code: '{result.ResultCode}'"
                                        },
                                        new TextContentBlock
                                        {
                                            Text = $"Output: '{result.Output}'"
                                        },
                                        new TextContentBlock
                                        {
                                            Text = $"Error: '{result.Error}'"
                                        }
                                    ]
                                };
                            }
                            catch (Exception e)
                            {
                                logger.LogError(e, "Failed to run code!");
                                return new CallToolResult
                                {
                                    Content =
                                    [
                                        new TextContentBlock
                                        {
                                            Text = $"Error: '{e.Message}'"
                                        }
                                    ]
                                };
                            }
                        }
                    }
                };
            })
            .WithHttpTransport();
    }

    public void Configure(WebApplication app)
    {
        app.UseDefaultFiles();
        app.UseStaticFiles();
        app.UseRouting();
        app.MapDefaultControllerRoute();
#pragma warning disable ASP0014
        app.UseCors("CorsPolicy");
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapMcp("/mcp");
        });
    }
}
