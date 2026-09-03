namespace Aiursoft.CppRunner.Configuration;

public class DockerImageSettings
{
    public const string SectionName = "DockerImageSettings";

    public string Prefix { get; init; } = string.Empty;

    public bool RequireAuthentication { get; init; }

    public string Username { get; init; } = string.Empty;

    public string Password { get; init; } = string.Empty;
}
