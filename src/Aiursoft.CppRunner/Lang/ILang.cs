namespace Aiursoft.CppRunner.Lang;

public interface ILang
{
    string LangName { get; set; }
    
    string LangExtension { get; set; }

    string DefaultCode { get; set; }

    string FileName { get; set; }

    string DockerImage { get; set; }

    string RunCommand { get; set; }
    
    Dictionary<string, string> OtherFiles { get; set; }
}