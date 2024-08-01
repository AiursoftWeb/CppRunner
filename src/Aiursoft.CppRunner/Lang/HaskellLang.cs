namespace Aiursoft.CppRunner.Lang;

public class HaskellLang : ILang
{
    public string LangDisplayName => "Haskell (GHC 9.8.1)";

    public string LangExtension => "haskell";

    public string LangName => "haskell";

    public string DefaultCode =>
        """
        fibonacci :: [Integer]
        fibonacci = 1 : 1 : zipWith (+) fibonacci (tail fibonacci)

        main :: IO ()
        main = print $ take 20 fibonacci
        """;

    public string EntryFileName => "main.hs";

    public string DockerImage => "haskell:9.8.1";

    public string RunCommand => "runhaskell /app/main.hs";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}