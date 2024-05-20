import type { Language } from "./models/language";

export type State = {
  languages: Language[];
  /**
   * language that written at CodeEditor
   */
  lang: string;
  /** 
   * highlight style name
   */
  langExtension: string;
  running: boolean;
};
