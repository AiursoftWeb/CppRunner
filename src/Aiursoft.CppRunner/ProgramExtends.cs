using Aiursoft.Canon;
using Aiursoft.CppRunner.Authorization;
using Aiursoft.CppRunner.Entities;
using Aiursoft.CSTools.Services;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using Aiursoft.CppRunner.Services;
using Aiursoft.CppRunner.Services.FileStorage;
using System.Diagnostics.CodeAnalysis;

namespace Aiursoft.CppRunner;

[ExcludeFromCodeCoverage]
public static class ProgramExtends
{
    [ExcludeFromCodeCoverage]
    private static async Task<bool> ShouldSeedAsync(TemplateDbContext dbContext)
    {
        var haveUsers = await dbContext.Users.AnyAsync();
        var haveRoles = await dbContext.Roles.AnyAsync();
        return !haveUsers && !haveRoles;
    }

    [ExcludeFromCodeCoverage]
    public static Task<IHost> CopyAvatarFileAsync(this IHost host)
    {
        using var scope = host.Services.CreateScope();
        var services = scope.ServiceProvider;
        var storageService = services.GetRequiredService<StorageService>();
        var logger = services.GetRequiredService<ILogger<Program>>();
        var avatarFilePath = Path.Combine(host.Services.GetRequiredService<IHostEnvironment>().ContentRootPath,
            "wwwroot", "images", "default-avatar.jpg");
        var physicalPath = storageService.GetFilePhysicalPath(User.DefaultAvatarPath);
        if (!File.Exists(avatarFilePath))
        {
            logger.LogWarning("Avatar file does not exist. Skip copying.");
            return Task.FromResult(host);
        }

        if (File.Exists(physicalPath))
        {
            logger.LogInformation("Avatar file already exists. Skip copying.");
            return Task.FromResult(host);
        }

        if (!Directory.Exists(Path.GetDirectoryName(physicalPath)))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(physicalPath)!);
        }

        File.Copy(avatarFilePath, physicalPath);
        logger.LogInformation("Avatar file copied to {Path}", physicalPath);
        return Task.FromResult(host);
    }

    [ExcludeFromCodeCoverage]
    public static async Task<IHost> SeedAsync(this IHost host)
    {
        using var scope = host.Services.CreateScope();
        var services = scope.ServiceProvider;
        var db = services.GetRequiredService<TemplateDbContext>();
        var logger = services.GetRequiredService<ILogger<Program>>();
        
        var settingsService = services.GetRequiredService<GlobalSettingsService>();
        await settingsService.SeedSettingsAsync();

        var shouldSeed = await ShouldSeedAsync(db);
        if (!shouldSeed)
        {
            logger.LogInformation("Do not need to seed the database. There are already users or roles present.");
            return host;
        }

        logger.LogInformation("Seeding the database with initial data...");
        var userManager = services.GetRequiredService<UserManager<User>>();
        var roleManager = services.GetRequiredService<RoleManager<IdentityRole>>();

        var role = await roleManager.FindByNameAsync("Administrators");
        if (role == null)
        {
            role = new IdentityRole("Administrators");
            await roleManager.CreateAsync(role);
        }

        var existingClaims = await roleManager.GetClaimsAsync(role);
        var existingClaimValues = existingClaims
            .Where(c => c.Type == AppPermissions.Type)
            .Select(c => c.Value)
            .ToHashSet();

        foreach (var permission in AppPermissions.GetAllPermissions())
        {
            if (!existingClaimValues.Contains(permission.Key))
            {
                var claim = new Claim(AppPermissions.Type, permission.Key);
                await roleManager.AddClaimAsync(role, claim);
            }
        }

        if (!await db.Users.AnyAsync(u => u.UserName == "admin"))
        {
            var user = new User
            {
                UserName = "admin",
                DisplayName = "Super Administrator",
                Email = "admin@default.com",
            };
            _ = await userManager.CreateAsync(user, "admin123");
            await userManager.AddToRoleAsync(user, "Administrators");
        }

        return host;
    }

    public static string GetDockerImagePullEndpoint(this ILang lang, string? prefixFromConfig)
    {
        var trimmedPrefix = string.Empty;
        if (!string.IsNullOrWhiteSpace(prefixFromConfig))
        {
            trimmedPrefix = prefixFromConfig.TrimEnd('/') + '/';
        }

        if (lang.DockerImage.StartsWith(trimmedPrefix))
        {
            return lang.DockerImage;
        }

        return trimmedPrefix + lang.DockerImage;
    }

    [ExcludeFromCodeCoverage]
    public static async Task PullContainersAsync(this IHost host)
    {
        using var scope = host.Services.CreateScope();
        var services = scope.ServiceProvider;
        var commandService = services.GetRequiredService<CommandService>();
        var configuration = services.GetRequiredService<IConfiguration>();
        var logger = services.GetRequiredService<ILogger<Program>>();
        var langs = services.GetRequiredService<IEnumerable<ILang>>();
        var hasGpuService = services.GetRequiredService<HasGpuService>();
        var retryEngine = services.GetRequiredService<RetryEngine>();
        var pool = services.GetRequiredService<CanonPool>();
        var prefix = configuration["DockerImageSettings:Prefix"];

        var downloadedImages = await commandService.RunCommandAsync("docker", "images", Path.GetTempPath());
        logger.LogInformation("Downloaded images count: {ImagesCount}", downloadedImages.output.Split('\n').Length);

        var hasGpu = await hasGpuService.HasNvidiaGpuForDockerWithCache();
        logger.LogInformation("Has GPU: {HasGpu}", hasGpu);

        var availableLangs = hasGpu
            ? langs
            : langs.Where(l => !l.NeedGpu);

        foreach (var lang in availableLangs)
        {
            if (downloadedImages.output.Contains(lang.GetDockerImagePullEndpoint(prefix)))
            {
                logger.LogInformation("Docker image {Image} already downloaded.", lang.GetDockerImagePullEndpoint(prefix));
                continue;
            }
            pool.RegisterNewTaskToPool(() => retryEngine.RunWithRetry(async _ =>
            {
                logger.LogInformation("Pulling docker image {Image}", lang.GetDockerImagePullEndpoint(prefix));
                var result = await commandService.RunCommandAsync("docker", $"pull {lang.GetDockerImagePullEndpoint(prefix)}", path: Path.GetTempPath(), timeout: TimeSpan.FromMinutes(10));
                if (result.code != 0)
                {
                    throw new Exception($"Failed to pull docker image {lang.GetDockerImagePullEndpoint(prefix)}! Error: {result.error}");
                }
            }, 5));
        }

        await pool.RunAllTasksInPoolAsync();
    }
}
