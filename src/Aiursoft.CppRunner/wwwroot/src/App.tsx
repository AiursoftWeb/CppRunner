import { MutableRefObject, useEffect, useRef, useState } from 'react'
import CodeEditor from '@uiw/react-textarea-code-editor';
import { getSupportLanguages, getDefaultCode, runCode } from './models/language';
import type { Language, OutputResult } from './models/language';
import './styles/tailwind.css';

function App() {

  useEffect(() => {
    const execute = async () => {
      const langs = await getSupportLanguages();
      setLanguages(langs);

      if (langs.length !== 0) {
        const selectLang = await getDefaultCode(langs[0].langName!);
        setCode(selectLang);
        setLang(langs[0].langName!);
      }
    };

    execute();
  }, []);

  const [languages, setLanguages]: [Language[], Function] = useState([]);
  const [data, setData] = useState('');
  const [code, setCode] = useState('');
  const [lang, setLang] = useState('');
  const [runing, setRuning] = useState(false);

  let langSelectRef: MutableRefObject<HTMLSelectElement | null> = useRef(null);

  const handleClear = () => {
    setData('');
  }

  const handleSelectLang = async (lang: string) => {
    const selectLang = await getDefaultCode(lang);
    setCode(selectLang);
    setLang(lang);
  }

  const handleRun = async () => {
    try {
      setRuning(true);
      const lang = langSelectRef.current!.value;
      const result: OutputResult = await runCode(lang, code);
      setData(result.output! || result.error!);
    } catch (error) {
      console.error(error);
    }
    finally {
      setRuning(false);
    }
  }

  return (
    <div className='flex flex-col h-screen'>
      <div className='w-screen h-[10vh] center text-lg'>
        Anduin's code background
      </div>
      <div className='flex flex-row gap-4 h-[85vh] w-full px-2'>
        <div className='flex flex-col flex-grow max-w-[50vw]'>
          <div className='flex items-center my-4 space-x-4'>
            <span>Input:</span>
            <select ref={langSelectRef} onChange={async (event) => { await handleSelectLang(event.target.value) }}>
              {
                Array.from(languages).map((lang) => (
                  <option key={lang.langName} value={lang.langName}>{lang.langDisplayName}</option>
                ))
              }
            </select>
            <span className='flex-grow flex flex-row-reverse'>
              <span><button className='p-2' disabled={runing} onClick={async () => { await handleRun() }}>Run</button></span>
            </span>
          </div>
          <div className='h-full overflow-scroll'>
            <div className='max-w-[50vw] h-full'>
              <CodeEditor
                value={code}
                language={lang}
                placeholder="Please enter CPP code."
                onChange={(evn) => setCode(evn.target.value)}
                padding={15}
                minHeight={100}
                style={{
                  backgroundColor: "#161b22",
                  minHeight: '100%',
                  fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
                }}
              />
            </div>
          </div>

        </div>

        <div className='flex-grow flex flex-col overflow-scroll max-w-[50vw]'>
          <div className='flex items-center my-4 pr-4'>
            <span>output:</span>
            <span className='flex-grow flex flex-row-reverse'>
              <button type='button' className='p-2' onClick={() => { handleClear() }}>Clear</button>
            </span>
          </div>
          <div className='relative h-full p-2 rounded border border-gray-600 break-words'>
            {runing && <div className='absolute bottom-1/2 left-1/2 z-50 center text-lg'>Runing...</div>}
            <CodeEditor
              value={data}
              language={lang}
              padding={15}
              minHeight={100}
              disabled={true}
              style={{
                backgroundColor: "#202b3c",
                minHeight: '100%',
                fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
              }}
            />
          </div>
        </div>
      </div>

      <footer className='center h-[5vh]'>
        Page Created by Dvorak
      </footer>
    </div>

  )
}

export default App
