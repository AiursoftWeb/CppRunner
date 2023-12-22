import type { Language } from "./models/language";

export type State = {
  languages: Language[];
  code: string;
  lang: string;
  running: boolean;
};