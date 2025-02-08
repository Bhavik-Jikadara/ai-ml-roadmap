import { ChevronRight, Github, Linkedin, Twitter, Mail, BookOpen, Heart, ExternalLink } from 'lucide-react';
import Medium from '../../assets/medium.png';

export const Footer = () => {
  const quickLinks = [
    { name: 'Get Started', href: '/', icon: <ChevronRight className="w-4 h-4" /> },
    { name: 'Resources', href: 'https://bhavikjikadara.medium.com', icon: <BookOpen className="w-4 h-4" /> },
    { name: 'Contact', href: 'mailto:bhavikjikadara33523@gmail.com', icon: <Mail className="w-4 h-4" /> },
  ];

  const socialLinks = [
    { name: 'GitHub', href: 'https://github.com/Bhavik-Jikadara', icon: <Github className="w-6 h-6" /> },
    { name: 'LinkedIn', href: 'https://www.linkedin.com/in/bhavik-jikadara', icon: <Linkedin className="w-6 h-6" /> },
    { name: 'Twitter', href: 'https://x.com/Bhavikjikadara1', icon: <Twitter className="w-6 h-6" /> },
  ];

  return (
    <footer className="relative bg-gradient-to-b from-gray-900 to-black text-white py-20 mt-24 border-t border-gray-800">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,_var(--tw-gradient-stops))] from-blue-500/10 via-purple-500/10 to-transparent" />

      <div className="max-w-6xl mx-auto px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 lg:gap-8">
          {/* Brand Section */}
          <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center space-x-3">
              <div className="w-20 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <span className="text-white font-bold text-xl">AI/ML</span>
              </div>
              <h3 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Roadmap
              </h3>
            </div>
            <p className="text-gray-300 text-lg leading-relaxed">
              Your comprehensive guide to mastering Artificial Intelligence and Machine Learning.
              Join our community of learners and start your journey today.
            </p>
            {/* Fixed the nested anchor tags issue */}
            <a
              href="https://bhavikjikadara.medium.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-6 py-3 rounded-lg text-sm font-medium transition-all transform hover:scale-105"
            >
              <img src={Medium} alt="Medium" className="w-6 h-6" />
              <span>Follow on Medium</span>
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          {/* Quick Links Section */}
          <div className="space-y-6">
            <h4 className="text-xl font-semibold text-white">Quick Links</h4>
            <ul className="space-y-4">
              {quickLinks.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel={link.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                    className="group flex items-center space-x-2 text-gray-300 hover:text-blue-400 transition-all"
                  >
                    <span className="transform transition-transform group-hover:translate-x-1">
                      {link.icon}
                    </span>
                    <span>{link.name}</span>
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Connect Section */}
          <div className="space-y-6">
            <h4 className="text-xl font-semibold text-white">Connect</h4>
            <div className="flex flex-col space-y-4">
              {socialLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-3 text-gray-300 hover:text-blue-400 transition-all transform hover:translate-x-1"
                >
                  {link.icon}
                  <span>{link.name}</span>
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Copyright Section */}
        <div className="mt-16 pt-8 border-t border-gray-800 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
          <p className="text-gray-400">&copy; 2024-25 AI/ML Roadmap. All rights reserved.</p>
          <div className="flex items-center space-x-2 text-gray-400">
            <span>Made with</span>
            <Heart className="w-4 h-4 text-red-500 animate-pulse" />
            <span>by Bhavik Jikadara</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;