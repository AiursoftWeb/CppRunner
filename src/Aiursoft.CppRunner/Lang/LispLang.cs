namespace Aiursoft.CppRunner.Lang;

public class LispLang : ILang
{
    public string LangDisplayName => "Lisp (rigetti/lisp)";

    public string LangExtension => "lisp";

    public string LangName => "lisp";

    public string DefaultCode =>
        """
        (defun fibonacci (n a b)
          (if (= n 0)
              a
              (fibonacci (- n 1) b (+ a b))))

        (dotimes (n 20)
          (format t "~d " (fibonacci (+ n) 1 1)))
        """;

    public string EntryFileName => "main.lisp";

    public string DockerImage => "rigetti/lisp:latest";

    public string RunCommand => "sbcl --script /app/main.lisp";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}
