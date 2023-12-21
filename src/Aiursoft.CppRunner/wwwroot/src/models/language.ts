/**
 * get support languages
 * @returns support languages
 */
export async function getSupportLanguages() {
  let supportLanguages: Language[] = await fetch(
    "https:cpprunner.aiursoft.cn/langs"
  ).then((resp) => {
    return resp.json();
  });

  return supportLanguages;
}

export async function getDefaultCode(lang: string) {
  let code = await fetch(
    `https:cpprunner.aiursoft.cn/langs/${lang}/default`
  ).then((resp) => {
    return resp.text();
  });

  return code;
}

export async function runCode(lang: string, code: string) {
  let result: OutputResult = await fetch(
    `https://cpprunner.aiursoft.cn/runner/run?lang=${lang}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: code,
    }
  ).then((resp) => resp.json());

  return result;
}

export class Language {
  langDisplayName?: string;
  langName?: string;
  langExtension?: string;
}

export class OutputResult {
  resultCode?: string;
  output?: string;
  error?: string;
}

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
