import './App.css'
import { Analytics } from "@vercel/analytics/react"
import Roadmap from './components/Roadmap/Roadmap'

function App() {
    return (
        <>
            <Roadmap />
            <Analytics />
        </>
    )
}

export default App
