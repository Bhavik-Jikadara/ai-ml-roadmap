import { ChevronRight, Github, Linkedin, Twitter } from 'lucide-react';

export const Footer = () => (
    <footer className="relative bg-gradient-to-b from-gray-900 to-black text-white py-16 mt-20 border-t border-gray-800">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-gray-800/10 via-transparent to-transparent" />
      <div className="max-w-6xl mx-auto px-6 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-16">
          <div className="space-y-4">
            <h3 className="text-3xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              AI/ML Roadmap
            </h3>
            <p className="text-gray-300 text-lg">
              Your comprehensive guide to mastering AI and Machine Learning
            </p>
          </div>
          <div className="space-y-4">
            <h4 className="text-2xl font-semibold mb-6 text-white">Quick Links</h4>
            <ul className="space-y-4 text-gray-300">
              <li>
                <a href="/" className="hover:text-blue-400 transition-colors flex items-center space-x-2">
                  <ChevronRight className="w-4 h-4" />
                  <span>Get Started</span>
                </a>
              </li>
              <li>
                <a href="https://bhavikjikadara.medium.com" className="hover:text-blue-400 transition-colors flex items-center space-x-2" target="_blank">
                  <ChevronRight className="w-4 h-4" />
                  <span>Resources</span>
                </a>
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <h4 className="text-2xl font-semibold mb-6 text-white">Connect</h4>
            <div className="flex space-x-6">
              <a href="#" className="text-gray-300 hover:text-blue-400 transition-colors transform hover:scale-110">
                <Github className="w-8 h-8" />
              </a>
              <a href="#" className="text-gray-300 hover:text-blue-400 transition-colors transform hover:scale-110">
                <Linkedin className="w-8 h-8" />
              </a>
              <a href="#" className="text-gray-300 hover:text-blue-400 transition-colors transform hover:scale-110">
                <Twitter className="w-8 h-8" />
              </a>
            </div>
          </div>
        </div>
        <div className="mt-16 pt-8 border-t border-gray-800 text-center">
          <p className="text-gray-400">&copy; 2024 AI/ML Roadmap. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );