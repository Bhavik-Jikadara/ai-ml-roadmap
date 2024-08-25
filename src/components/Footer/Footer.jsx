import './Footer.css'; // Ensure this file exists with proper styles
import 'bootstrap-icons/font/bootstrap-icons.css';

function Footer() {
    return (
        <footer className="footer">
            <div className="footer-container">
                <div className='footer-top'>
                    <h2>AI/ML Roadmap</h2>
                    <div className="footer-section logo-social">
                        <div className="social-media">
                            <a href="https://www.linkedin.com/in/bhavikjikadara" target="_blank" rel="noopener noreferrer">
                                <i className="bi bi-linkedin"></i>
                            </a>
                            <a href="https://github.com/Bhavik-Jikadara" target="_blank" rel="noopener noreferrer">
                                <i className="bi bi-github"></i>
                            </a>
                            <a href="https://www.facebook.com/Bhavikjikadara07" target="_blank" rel="noopener noreferrer">
                                <i className="bi bi-facebook"></i>
                            </a>
                            <a href="https://www.instagram.com/bhavikjikadara/" target="_blank" rel="noopener noreferrer">
                                <i className="bi bi-instagram"></i>
                            </a>
                            <a href="https://twitter.com/BhavikJikadara1" target="_blank" rel="noopener noreferrer">
                                <i className="bi bi-twitter"></i>
                            </a>
                            <a href="https://bhavikjikadara.medium.com/" target="_blank" rel="noopener noreferrer">
                                <i className="bi bi-medium"></i>
                            </a>
                        </div>
                    </div>
                </div> 
            </div>
            <div className="footer-bottom">
                <small>&copy; 2024 AI/ML Roadmap by <span><a href="https://www.linkedin.com/in/bhavikjikadara">Bhavik Jikadara</a></span>. All rights reserved.</small>
            </div>
        </footer>
    );
}

export default Footer;
