import { useState } from 'react';
import { ChevronRight, ExternalLink } from 'lucide-react';
import PropTypes from 'prop-types';



export const Header = () => (
    <div className="relative overflow-hidden mb-12 mt-12 rounded-xl bg-gray-900/80 backdrop-blur-sm shadow-xl">
      <div className="absolute inset-0 bg-gradient-to-r from-gray-900 to-blue-900 opacity-90" />
      <div className="relative bg-gradient-to-r from-gray-900 to-blue-900 p-12 rounded-b-3xl shadow-2xl">
        <div className="max-w-4xl mx-auto relative">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-white/5 to-transparent animate-pulse" />
          <h1 className="text-6xl font-bold mb-8 text-white bg-clip-text relative z-10 animate-fade-in">
            AI/ML Roadmap 2024
          </h1>
          <p className="text-xl text-gray-200 leading-relaxed max-w-2xl relative z-10 animate-slide-up">
            Master Artificial Intelligence and Machine Learning with our comprehensive learning path.
            Access curated resources and join a community of learners.
          </p>
        </div>
      </div>
    </div>
  );

export const Checkpoint = ({ title, items, index, isVisible }) => {
    const [isExpanded, setIsExpanded] = useState(false);
  
    return (
      <div 
        className={`transform transition-all duration-700 ${
          isVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
        }`}
      >
        <div className="bg-gray-900/80 backdrop-blur-sm rounded-xl p-6 mb-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-800">
          <div 
            className="flex items-center justify-between mb-6 cursor-pointer"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center">
              <div className="w-10 h-10 bg-blue-900 rounded-full flex items-center justify-center mr-4">
                <span className="text-white font-bold">{index + 1}</span>
              </div>
              <h2 className="text-2xl font-bold text-white">{title}</h2>
            </div>
            <ChevronRight 
              className={`w-6 h-6 text-gray-400 transform transition-transform duration-300 ${
                isExpanded ? 'rotate-90' : ''
              }`}
            />
          </div>
          
          <div className={`space-y-4 overflow-hidden transition-all duration-500 ${
            isExpanded ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
          }`}>
            {items.map((item, idx) => (
              <div 
                key={idx}
                className="flex items-center bg-gray-800/50 backdrop-blur-sm p-4 rounded-lg hover:bg-gray-700/50 transition-colors group"
              >
                {item.href ? (
                  <a
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-200 hover:text-blue-400 flex items-center justify-between w-full group"
                  >
                    <span>{item.text || item}</span>
                    <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                ) : (
                  <span className="text-gray-200">{item.text || item}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };
