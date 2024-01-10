import type { Language } from "./models/language";

export type State = {
  languages: Language[];
  lang: string;
  running: boolean;
};