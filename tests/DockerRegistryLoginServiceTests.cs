using Aiursoft.CppRunner.Configuration;
using Aiursoft.CppRunner.Services;
using Microsoft.Extensions.Logging.Abstractions;

namespace Aiursoft.CppRunner.Tests;

[TestClass]
public class DockerRegistryLoginServiceTests
{
    [TestMethod]
    [DataRow("hub.aiursoft.com/", "hub.aiursoft.com")]
    [DataRow("hub.aiursoft.com/public_mirror/", "hub.aiursoft.com")]
    [DataRow("localhost:8080/public_mirror", "localhost:8080")]
    public void ExtractsRegistryFromImagePrefix(string prefix, string expected)
    {
        Assert.AreEqual(expected, DockerRegistryLoginService.GetRegistryFromPrefix(prefix));
    }

    [TestMethod]
    public async Task AuthenticationDisabledDoesNotStartDocker()
    {
        var service = new DockerRegistryLoginService(NullLogger<DockerRegistryLoginService>.Instance);

        await service.LoginIfRequiredAsync(
            new DockerImageSettings { RequireAuthentication = false },
            dockerExecutable: "/this/docker-command-does-not-exist");
    }

    [TestMethod]
    public async Task PasswordIsPassedThroughStandardInput()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }

        var tempDirectory = Path.Combine(Path.GetTempPath(), $"cpprunner-docker-login-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDirectory);
        var fakeDocker = Path.Combine(tempDirectory, "docker");
        var argumentsFile = fakeDocker + ".arguments";
        var passwordFile = fakeDocker + ".password";

        try
        {
            await File.WriteAllTextAsync(fakeDocker, $$"""
                #!/bin/sh
                printf '%s\n' "$@" > '{{argumentsFile}}'
                cat > '{{passwordFile}}'
                printf 'Login Succeeded\n'
                """);
            File.SetUnixFileMode(
                fakeDocker,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);

            var service = new DockerRegistryLoginService(NullLogger<DockerRegistryLoginService>.Instance);
            await service.LoginIfRequiredAsync(
                new DockerImageSettings
                {
                    Prefix = "hub.aiursoft.com/public_mirror/",
                    RequireAuthentication = true,
                    Username = "registry-user",
                    Password = "top-secret"
                },
                fakeDocker);

            var arguments = await File.ReadAllLinesAsync(argumentsFile);
            CollectionAssert.AreEqual(
                new[] { "login", "--username", "registry-user", "--password-stdin", "hub.aiursoft.com" },
                arguments);
            Assert.AreEqual($"top-secret{Environment.NewLine}", await File.ReadAllTextAsync(passwordFile));
            Assert.IsFalse(string.Join(' ', arguments).Contains("top-secret", StringComparison.Ordinal));
        }
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }
}
