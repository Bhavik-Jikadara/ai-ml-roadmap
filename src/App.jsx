import './App.css'
import { Analytics } from "@vercel/analytics/react"
import Footer from './components/Footer/Footer'
import Header from './components/Header/Header'
import Content from './components/Content/Content'
import { SpeedInsights } from "@vercel/speed-insights/next"

function App() {
    return (
        <>
            <Header />
            <hr></hr>
            <Content />
            <Footer />
            <Analytics />
            <SpeedInsights />
        </>
    )
}

export default App
