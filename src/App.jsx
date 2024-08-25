import './App.css'
import { Analytics } from "@vercel/analytics/react"
import Footer from './components/Footer/Footer'
import Header from './components/Header/Header'
import Content from './components/Content/Content'

function App() {
    return (
        <>
            <Header />
            <hr></hr>
            <Content />
            <Footer />
            <Analytics />
        </>
    )
}

export default App
