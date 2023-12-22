namespace Aiursoft.CppRunner.Lang;

public class HaskellLang : ILang
{
    public string LangDisplayName { get; set; } = "Haskell (GHC 9.8.1)";

    public string LangExtension { get; set; } = "hs";

    public string LangName { get; set; } = "haskell";

    public string DefaultCode { get; set; } =
        """
        fibonacci :: [Integer]
        fibonacci = 1 : 1 : zipWith (+) fibonacci (tail fibonacci)
        
        main :: IO ()
        main = print $ take 20 fibonacci
        """;

    public string EntryFileName { get; set; } = "main.hs";

    public string DockerImage { get; set; } = "haskell:9.8.1";

    public string RunCommand { get; set; } = "runhaskell /app/main.hs";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}