/**
 * get support languages
 * @returns support languages
 */
export async function getSupportLanguages() {
  const supportLanguages: Language[] = await fetch("/langs").then((resp) => {
    return resp.json();
  });

  return supportLanguages;
}

const defaultCodeCache = new Map<string, string>();

export async function getDefaultCode(lang: string): Promise<string> {
  if (defaultCodeCache.has(lang)) {
    return defaultCodeCache.get(lang) as string;
  }

  const code = await fetch(`/langs/${lang}/default`).then((resp) => {
    return resp.text();
  });

  defaultCodeCache.set(lang, code);
  return code;
}

export function runCode(
  lang: string,
  code: string
): [Promise<OutputResult>, AbortController] {
  const controller = new AbortController();

  const promise = fetch(`/runner/run?lang=${lang}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: code,
    signal: controller.signal,
  })
    .then((resp) => {
      if (resp.ok) {
        return resp.json();
      } else {
        throw mapError(resp.status, resp.statusText);
      }
    })
    .catch((e) => {
      if (e.name == "AbortError") {
        return {} as OutputResult;
      }
      throw e;
    });

  return [promise, controller];
}

function mapError(status: number, text: string): Error {
  switch (status) {
    case 429:
      throw "Too Many Request! try again in 2 minutes later";
    default:
      throw `${status} ${text}`;
  }
}

export type Language = {
  langDisplayName?: string;
  langName?: string;
  langExtension?: string;
};

export type OutputResult = {
  resultCode?: number;
  output?: string;
  error?: string;
};

/*
"abap"
"aes"
"apex"
"azcli"
"bat"
"bicep"
"brainfuck"
"c"
"cameligo"
"clike"
"clojure"
"coffeescript"
"cpp"
"csharp"
"csp"
"css"
"dart"
"dockerfile"
"ecl"
"elixir"
"erlang"
"flow9"
"freemarker2"
"fsharp"
"go"
"graphql"
"handlebars"
"hcl"
"html"
"ini"
"java"
"javascript"
"json"
"jsx"
"julia"
"kotlin"
"less"
"lex"
"lexon"
"liquid"
"livescript"
"lua"
"m3"
"markdown"
"mips"
"msdax"
"mysql"
"nginx"
"pascal"
"pascaligo"
"perl"
"pgsql"
"php"
"pla"
"plaintext"
"postiats"
"powerquery"
"powershell"
"proto"
"pug"
"python"
"qsharp"
"r"
"razor"
"redis"
"redshift"
"restructuredtext"
"ruby"
"rust"
"sb"
"scala"
"scheme"
"scss"
"shell"
"sol"
"sparql"
"sql"
"st"
"stylus"
"swift"
"systemverilog"
"tcl"
"toml"
"tsx"
"twig"
"typescript"
"vb"
"vbscript"
"verilog"
"vue"
"xml"
"yaml"
*/
