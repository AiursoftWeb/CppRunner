namespace Aiursoft.CppRunner;

public interface ILang
{
    string LangDisplayName { get; }

    string LangName { get; }
    
    string LangExtension { get; }

    string DefaultCode { get; }

    string EntryFileName { get; }

    string DockerImage { get; }

    string RunCommand { get; }
    
    bool NeedGpu{ get; }
    
    Dictionary<string, string> OtherFiles { get; }
}