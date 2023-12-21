namespace Aiursoft.CppRunner.Lang;

public class LispLang : ILang
{
    public string LangDisplayName { get; set; } = "Lisp (SBCL 2.1)";

    public string LangExtension { get; set; } = "lisp";

    public string LangName { get; set; } = "lisp";

    public string DefaultCode { get; set; } =
        """
        (defun fibonacci (n a b)
          (if (= n 0)
              a
              (fibonacci (- n 1) b (+ a b))))

        (dotimes (n 20)
          (format t "~d " (fibonacci n 0 1)))
        """;

    public string EntryFileName { get; set; } = "main.lisp";

    public string DockerImage { get; set; } = "sbcl:2.1-alpine";

    public string RunCommand { get; set; } = "sbcl --script /app/main.lisp";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}