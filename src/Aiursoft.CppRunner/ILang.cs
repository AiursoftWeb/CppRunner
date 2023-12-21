namespace Aiursoft.CppRunner;

public interface ILang
{
    string LangDisplayName { get; set; }

    string LangName { get; set; }
    
    string LangExtension { get; set; }

    string DefaultCode { get; set; }

    string EntryFileName { get; set; }

    string DockerImage { get; set; }

    string RunCommand { get; set; }
    
    Dictionary<string, string> OtherFiles { get; set; }
}