using System.Text.Json;
using Aiursoft.Canon;
using Aiursoft.CSTools.Tools;
using Aiursoft.DbTools.Switchable;
using Aiursoft.Scanner;
using Aiursoft.CppRunner.Configuration;
using Aiursoft.WebTools.Abstractions.Models;
using Aiursoft.CppRunner.InMemory;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CppRunner.MySql;
using Aiursoft.CppRunner.Services.Authentication;
using Aiursoft.CppRunner.Sqlite;
using Aiursoft.UiStack.Layout;
using Aiursoft.UiStack.Navigation;
using Microsoft.AspNetCore.Mvc.Razor;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using System.Diagnostics.CodeAnalysis;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner;

[ExcludeFromCodeCoverage]
public class Startup : IWebStartup
{
    public void ConfigureServices(IConfiguration configuration, IWebHostEnvironment environment, IServiceCollection services)
    {
        // AppSettings.
        services.Configure<AppSettings>(configuration.GetSection("AppSettings"));

        // Relational database
        var (connectionString, dbType, allowCache) = configuration.GetDbSettings();
        services.AddSwitchableRelationalDatabase(
            dbType: EntryExtends.IsInUnitTests() ? "InMemory" : dbType,
            connectionString: connectionString,
            supportedDbs:
            [
                new MySqlSupportedDb(allowCache: allowCache, splitQuery: false),
                new SqliteSupportedDb(allowCache: allowCache, splitQuery: true),
                new InMemorySupportedDb()
            ]);

        // Authentication and Authorization
        services.AddTemplateAuth(configuration);

        // Services
        services.AddMemoryCache();
        services.AddHttpClient();
        services.AddAssemblyDependencies(typeof(Startup).Assembly);
        services.AddSingleton<NavigationState<Startup>>();

        // Background job queue
        services.AddSingleton<Services.BackgroundJobs.BackgroundJobQueue>();
        services.AddHostedService<Services.BackgroundJobs.QueueWorkerService>();

        // Code Runner services
        services.AddScoped<CommandService>();
        services.AddScoped<Services.RunCodeService>();
        services.AddScoped<Services.HasGpuService>();
        services.AddTaskCanon();

        // Languages
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

        // MCP Server
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
                        ListChanged = true
                    }
                };
            })
            .WithHttpTransport(options => options.Stateless = true)
            .WithListToolsHandler((request, _) =>
            {
                var langs = request.Services!.GetRequiredService<IEnumerable<ILang>>();
                var logger = request.Services!.GetRequiredService<ILogger<Startup>>();
                logger.LogInformation("List tools...");
                var tools = langs.Select(l => new Tool
                    {
                        Name = $"run_{l.LangName}",
                        Description = $"Compile and run {l.LangDisplayName} code and get output and error results. No external libraries are supported. No user input support.",
                        InputSchema = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(@"
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
            })
            .WithCallToolHandler(async (request, _) =>
            {
                var runCodeService = request.Services!.GetRequiredService<Services.RunCodeService>();
                var logger = request.Services!.GetRequiredService<ILogger<Services.RunCodeService>>();
                var langs = request.Services!.GetRequiredService<IEnumerable<ILang>>();
                var toolName = request.Params?.Name
                               ?? throw new McpException("Missing tool name");

                logger.LogInformation("Call tool {ToolName}...", toolName);
                var langKey = toolName["run_".Length..];
                var langImpl = langs.FirstOrDefault(l =>
                                   l.LangName.Equals(langKey, StringComparison.OrdinalIgnoreCase))
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
            });

        // Controllers and localization
        services.AddControllersWithViews()
            .AddNewtonsoftJson(options =>
            {
                options.SerializerSettings.DateTimeZoneHandling = DateTimeZoneHandling.Utc;
                options.SerializerSettings.ContractResolver = new DefaultContractResolver();
            })
            .AddApplicationPart(typeof(Startup).Assembly)
            .AddApplicationPart(typeof(UiStackLayoutViewModel).Assembly)
            .AddViewLocalization(LanguageViewLocationExpanderFormat.Suffix)
            .AddDataAnnotationsLocalization();
    }

    public void Configure(WebApplication app)
    {
        app.UseExceptionHandler("/Error/Code500");
        app.UseStatusCodePagesWithReExecute("/Error/Code{0}");
        app.UseStaticFiles();
        app.UseRouting();
        app.UseAuthentication();
        app.UseAuthorization();
        app.MapDefaultControllerRoute();
        app.MapMcp("/mcp");
    }
}
