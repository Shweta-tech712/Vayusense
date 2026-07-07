import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown } from 'lucide-react';

export default function ScrollIndicator({ isTabActive }) {
  return (
    <motion.div
      animate={isTabActive ? {
        y: [0, 8, 0],
      } : {}}
      transition={{
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }}
      className="flex flex-col items-center space-y-1.5 cursor-pointer text-slate-400 hover:text-white transition-colors duration-300 select-none mt-6"
    >
      <span className="text-[10px] font-mono tracking-widest uppercase">Scroll to explore</span>
      <ArrowDown className="w-4 h-4 text-sky-400" />
    </motion.div>
  );
}
