import { useState, useEffect } from 'react';
import { Header } from '@/components/Header/Header';
import { Navbar } from '@/components/Header/Navbar';
import { Footer } from '@/components/Footer/Footer';
import Checkpoint from '@/components/Roadmap/Checkpoints';
import checkpointsData from '@/assets/checkpoints.json';

const RoadmapApp = () => {
  const [visibleCheckpoints, setVisibleCheckpoints] = useState(new Set());
  const [checkpoints, setCheckpoints] = useState([]);

  useEffect(() => {
    // Extract checkpoints array from the imported JSON
    setCheckpoints(checkpointsData || []);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const index = parseInt(entry.target.dataset.index);
            if (!Number.isNaN(index)) {
              setVisibleCheckpoints(prev => new Set([...prev, index]));
            }
          }
        });
      },
      { threshold: 0.1 }
    );

    const checkpointElements = document.querySelectorAll('.checkpoint-container');
    checkpointElements.forEach(checkpoint => {
      observer.observe(checkpoint);
    });

    return () => {
      checkpointElements.forEach(checkpoint => {
        observer.unobserve(checkpoint);
      });
    };
  }, [checkpoints]); // Add checkpoints as dependency

  // Guard clause for empty checkpoints
  if (!checkpoints.length) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900">
        <Navbar />
        <main className="max-w-6xl mx-auto px-4 md:px-6 lg:px-8 pt-16">
          <Header />
          <div className="flex justify-center items-center min-h-[400px]">
            <div className="text-gray-400">Loading roadmap...</div>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900">
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 md:px-6 lg:px-8 pt-16">
        <Header />
        <div className="max-w-4xl mx-auto">
          {checkpoints.map((checkpoint, index) => (
            <div 
              key={index} 
              className="checkpoint-container" 
              data-index={index}
            >
              <Checkpoint
                index={index}
                title={checkpoint.title}
                items={checkpoint.items}
                isVisible={visibleCheckpoints.has(index)}
              />
            </div>
          ))}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default RoadmapApp;