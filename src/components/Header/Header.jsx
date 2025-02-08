import { ExternalLink } from 'lucide-react';
import Medium from '../../assets/medium.png';

export const Header = () => (
  <div className="relative overflow-hidden mb-16 mt-24 rounded-2xl">
    <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 blur-3xl" />
    <div className="relative bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 p-12 md:p-16 rounded-2xl border border-gray-800/50 shadow-2xl">
      <div className="max-w-4xl mx-auto relative">
        <div className="absolute inset-0">
          <div className="absolute top-0 right-0 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-0 left-0 w-72 h-72 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-700" />
        </div>

        <div className="relative z-10 space-y-8">
          <div className="space-y-4">
            <div className="flex items-center space-x-3 mb-6">
              <span className="px-4 py-1 bg-blue-600/20 text-blue-400 rounded-full text-sm font-medium">
                2025 Edition
              </span>
              <span className="px-4 py-1 bg-purple-600/20 text-purple-400 rounded-full text-sm font-medium">
                Updated Weekly
              </span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              AI/ML Roadmap 2025
            </h1>
            <p className="text-xl text-gray-300 leading-relaxed max-w-2xl">
              Master Artificial Intelligence and Machine Learning with our comprehensive learning path.
              Access curated resources and join a community of learners.
            </p>
          </div>

          <div className="flex flex-wrap gap-6">
            <a href="https://bhavikjikadara.medium.com" target="_blank" className="flex items-center space-x-2 px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-xl font-medium transition-all border border-gray-700">
              <img src={Medium} alt="Medium" className="w-6 h-6" />
              <span>Follow on Medium</span>
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
);