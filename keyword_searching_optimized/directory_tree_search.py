#!/usr/bin/env python3
"""
Optimized Directory Tree Search Script for Large Directories (1TB+)

This script builds an optimized tree-based structure of directories and files,
then searches for files containing a specified keyword using parallel processing
and memory-efficient techniques.

Author: Aman Kumar
Date: 2025
"""

import os
import logging
import argparse
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Generator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import mimetypes
import chardet
import multiprocessing as mp
from collections import defaultdict
import gc
import psutil
import fnmatch
from functools import lru_cache


class SearchResult(Enum):
    """Enum for different types of search results"""
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class FileInfo:
    """Lightweight data class to store essential file information"""
    path: str  # Using string instead of Path for memory efficiency
    size: int
    modified_time: float
    search_result: SearchResult = SearchResult.NOT_FOUND
    error_message: Optional[str] = None


@dataclass
class SearchStats:
    """Statistics for the search operation"""
    total_files: int = 0
    files_searched: int = 0
    files_skipped: int = 0
    files_matched: int = 0
    files_error: int = 0
    directories_processed: int = 0
    bytes_processed: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_runtime(self) -> float:
        return time.time() - self.start_time


class OptimizedTreeBuilder:
    """Memory-efficient directory tree builder using generators"""
    
    def __init__(self, logger: logging.Logger, max_depth: int = 50):
        self.logger = logger
        self.max_depth = max_depth
        self.stats = SearchStats()
        
        # Pre-compiled file patterns for faster matching
        self.binary_patterns = {
            '*.exe', '*.bin', '*.dll', '*.so', '*.dylib',
            '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.ico',
            '*.mp3', '*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv',
            '*.zip', '*.tar', '*.gz', '*.rar', '*.7z', '*.bz2',
            '*.pdf', '*.doc', '*.docx', '*.xls', '*.xlsx', '*.ppt', '*.pptx',
            '*.sqlite', '*.db', '*.mdb'
        }
        
        # Text file patterns that are safe to search
        self.text_patterns = {
            '*.txt', '*.py', '*.js', '*.html', '*.htm', '*.css', '*.json',
            '*.xml', '*.yml', '*.yaml', '*.md', '*.rst', '*.log',
            '*.c', '*.cpp', '*.h', '*.hpp', '*.java', '*.cs', '*.php',
            '*.rb', '*.go', '*.rs', '*.swift', '*.kt', '*.scala',
            '*.sql', '*.sh', '*.bat', '*.ps1', '*.cfg', '*.conf', '*.ini'
        }
    
    def scan_directory_fast(self, root_path: Path, include_patterns: Optional[List[str]] = None,
                           exclude_patterns: Optional[List[str]] = None) -> Generator[FileInfo, None, None]:
        """
        Fast directory scanning using os.scandir for optimal performance
        
        Args:
            root_path: Root directory to scan
            include_patterns: File patterns to include (e.g., ['*.py', '*.txt'])
            exclude_patterns: Directory patterns to exclude (e.g., ['node_modules', '.git'])
            
        Yields:
            FileInfo objects for files that match criteria
        """
        exclude_patterns = exclude_patterns or [
            '.git', '.svn', '.hg', 'node_modules', '__pycache__', '.pytest_cache',
            '.venv', 'venv', '.env', 'build', 'dist', '.tox', '.mypy_cache'
        ]
        
        try:
            for file_info in self._scan_recursive(root_path, 0, include_patterns, exclude_patterns):
                yield file_info
                # Periodic garbage collection for very large directories
                if self.stats.total_files % 10000 == 0:
                    gc.collect()
                    
        except Exception as e:
            self.logger.error(f"Error scanning directory {root_path}: {e}")
    
    def _scan_recursive(self, path: Path, depth: int, include_patterns: Optional[List[str]],
                       exclude_patterns: List[str]) -> Generator[FileInfo, None, None]:
        """Recursive directory scanning with depth limiting"""
        
        if depth > self.max_depth:
            self.logger.warning(f"Max depth reached, skipping: {path}")
            return
        
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            # Quick file filtering
                            if self._should_process_file(entry.name, include_patterns):
                                stat_result = entry.stat()
                                file_info = FileInfo(
                                    path=entry.path,
                                    size=stat_result.st_size,
                                    modified_time=stat_result.st_mtime
                                )
                                self.stats.total_files += 1
                                yield file_info
                        
                        elif entry.is_dir(follow_symlinks=False):
                            # Skip excluded directories
                            if not any(fnmatch.fnmatch(entry.name.lower(), pattern.lower()) 
                                     for pattern in exclude_patterns):
                                self.stats.directories_processed += 1
                                yield from self._scan_recursive(
                                    Path(entry.path), depth + 1, include_patterns, exclude_patterns
                                )
                            else:
                                self.logger.debug(f"Skipping excluded directory: {entry.path}")
                                
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Cannot access {entry.path}: {e}")
                        continue
                        
        except (OSError, PermissionError) as e:
            self.logger.error(f"Cannot read directory {path}: {e}")
    
    @lru_cache(maxsize=1000)
    def _should_process_file(self, filename: str, include_patterns: Optional[Tuple[str, ...]]) -> bool:
        """Fast file filtering with caching"""
        filename_lower = filename.lower()
        
        # Skip hidden files and common non-text files
        if filename_lower.startswith('.') and not filename_lower.endswith(('.txt', '.log')):
            return False
        
        # If include patterns specified, check against them
        if include_patterns:
            return any(fnmatch.fnmatch(filename_lower, pattern.lower()) 
                      for pattern in include_patterns)
        
        # Skip obvious binary files
        if any(fnmatch.fnmatch(filename_lower, pattern) for pattern in self.binary_patterns):
            return False
        
        # Prefer known text files
        if any(fnmatch.fnmatch(filename_lower, pattern) for pattern in self.text_patterns):
            return True
        
        # For unknown extensions, include if small enough
        try:
            file_path = Path(filename)
            if file_path.suffix == '' or len(file_path.suffix) > 10:
                return False  # No extension or very long extension
            return True
        except:
            return False


class ParallelKeywordSearcher:
    """High-performance parallel keyword searcher"""
    
    def __init__(self, logger: logging.Logger, case_sensitive: bool = False, 
                 max_workers: Optional[int] = None, chunk_size: int = 100):
        self.logger = logger
        self.case_sensitive = case_sensitive
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit for individual files
        self.max_memory_usage = 0.8  # 80% of available memory
        
        # Pre-compile keyword for faster searching
        self.search_keyword = None
        
        # Memory monitoring
        self.process = psutil.Process()
        
    def search_files_parallel(self, file_generator: Generator[FileInfo, None, None], 
                            keyword: str) -> Tuple[List[str], SearchStats]:
        """
        Search for keyword in files using parallel processing
        
        Args:
            file_generator: Generator yielding FileInfo objects
            keyword: Keyword to search for
            
        Returns:
            Tuple of (matching file paths, search statistics)
        """
        self.search_keyword = keyword if self.case_sensitive else keyword.lower()
        matching_files = []
        stats = SearchStats()
        
        # Process files in chunks to manage memory
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            file_chunk = []
            futures = []
            
            for file_info in file_generator:
                file_chunk.append(file_info)
                
                # Process chunk when it reaches target size or memory limit
                if (len(file_chunk) >= self.chunk_size or 
                    self._should_process_chunk()):
                    
                    # Submit chunk for processing
                    future = executor.submit(self._process_file_chunk, file_chunk.copy())
                    futures.append(future)
                    file_chunk.clear()
                    
                    # Collect completed results to free memory
                    self._collect_completed_futures(futures, matching_files, stats)
            
            # Process remaining files
            if file_chunk:
                future = executor.submit(self._process_file_chunk, file_chunk)
                futures.append(future)
            
            # Collect all remaining results
            for future in as_completed(futures):
                chunk_results, chunk_stats = future.result()
                matching_files.extend(chunk_results)
                self._merge_stats(stats, chunk_stats)
        
        return matching_files, stats
    
    def _process_file_chunk(self, file_chunk: List[FileInfo]) -> Tuple[List[str], SearchStats]:
        """Process a chunk of files in parallel"""
        matching_files = []
        stats = SearchStats()
        
        with ThreadPoolExecutor(max_workers=min(8, len(file_chunk))) as executor:
            # Submit all files in chunk
            future_to_file = {
                executor.submit(self._search_single_file, file_info): file_info
                for file_info in file_chunk
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    if result == SearchResult.FOUND:
                        matching_files.append(file_info.path)
                        stats.files_matched += 1
                    elif result == SearchResult.SKIPPED:
                        stats.files_skipped += 1
                    elif result == SearchResult.ERROR:
                        stats.files_error += 1
                    else:
                        stats.files_searched += 1
                        
                    stats.bytes_processed += file_info.size
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_info.path}: {e}")
                    stats.files_error += 1
        
        return matching_files, stats
    
    def _search_single_file(self, file_info: FileInfo) -> SearchResult:
        """Search for keyword in a single file with optimizations"""
        file_path = Path(file_info.path)
        
        try:
            # Skip large files
            if file_info.size > self.max_file_size:
                return SearchResult.SKIPPED
            
            # Skip empty files
            if file_info.size == 0:
                return SearchResult.SKIPPED
            
            # Fast binary detection using file extension and magic bytes
            if self._is_likely_binary(file_path):
                return SearchResult.SKIPPED
            
            # Read file with optimized approach
            return self._read_and_search_optimized(file_path, file_info.size)
            
        except Exception as e:
            self.logger.debug(f"Error searching {file_path}: {e}")
            return SearchResult.ERROR
    
    def _read_and_search_optimized(self, file_path: Path, file_size: int) -> SearchResult:
        """Optimized file reading and searching"""
        try:
            # For small files, read entirely
            if file_size < 64 * 1024:  # 64KB
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Quick binary check
                if b'\x00' in data[:1024]:
                    return SearchResult.SKIPPED
                
                try:
                    text = data.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    return SearchResult.SKIPPED
                
                search_text = text if self.case_sensitive else text.lower()
                return SearchResult.FOUND if self.search_keyword in search_text else SearchResult.NOT_FOUND
            
            # For larger files, use streaming search
            return self._streaming_search(file_path)
            
        except Exception:
            return SearchResult.ERROR
    
    def _streaming_search(self, file_path: Path) -> SearchResult:
        """Memory-efficient streaming search for large files"""
        buffer_size = 64 * 1024  # 64KB buffer
        overlap_size = len(self.search_keyword.encode('utf-8')) * 2
        
        try:
            with open(file_path, 'rb') as f:
                previous_chunk = b''
                
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    
                    # Check for binary content
                    if b'\x00' in chunk[:1024] and not previous_chunk:
                        return SearchResult.SKIPPED
                    
                    # Combine with previous chunk overlap
                    search_chunk = previous_chunk + chunk
                    
                    try:
                        text = search_chunk.decode('utf-8', errors='ignore')
                        search_text = text if self.case_sensitive else text.lower()
                        
                        if self.search_keyword in search_text:
                            return SearchResult.FOUND
                        
                        # Keep overlap for next iteration
                        previous_chunk = chunk[-overlap_size:] if len(chunk) > overlap_size else chunk
                        
                    except UnicodeDecodeError:
                        return SearchResult.SKIPPED
                
                return SearchResult.NOT_FOUND
                
        except Exception:
            return SearchResult.ERROR
    
    @lru_cache(maxsize=500)
    def _is_likely_binary(self, file_path: Path) -> bool:
        """Fast binary file detection"""
        # Check extension first
        suffix = file_path.suffix.lower()
        binary_extensions = {
            '.exe', '.bin', '.dll', '.so', '.dylib', '.o', '.a',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.tiff',
            '.mp3', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.sqlite', '.db', '.mdb'
        }
        
        return suffix in binary_extensions
    
    def _should_process_chunk(self) -> bool:
        """Check if we should process current chunk based on memory usage"""
        try:
            memory_percent = self.process.memory_percent()
            return memory_percent > self.max_memory_usage * 100
        except:
            return False
    
    def _collect_completed_futures(self, futures: List, matching_files: List[str], stats: SearchStats):
        """Collect completed futures to free memory"""
        completed = [f for f in futures if f.done()]
        for future in completed:
            try:
                chunk_results, chunk_stats = future.result()
                matching_files.extend(chunk_results)
                self._merge_stats(stats, chunk_stats)
            except Exception as e:
                self.logger.error(f"Error collecting future result: {e}")
            futures.remove(future)
    
    def _merge_stats(self, target_stats: SearchStats, source_stats: SearchStats):
        """Merge search statistics"""
        target_stats.files_searched += source_stats.files_searched
        target_stats.files_skipped += source_stats.files_skipped
        target_stats.files_matched += source_stats.files_matched
        target_stats.files_error += source_stats.files_error
        target_stats.bytes_processed += source_stats.bytes_processed


class ProgressMonitor:
    """Real-time progress monitoring for large operations"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 5  # seconds
        
    def update_progress(self, stats: SearchStats, current_file: str = ""):
        """Update progress information"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            elapsed = current_time - self.start_time
            
            # Calculate rates
            files_per_sec = stats.total_files / elapsed if elapsed > 0 else 0
            mb_per_sec = (stats.bytes_processed / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            
            self.logger.info(
                f"Progress: {stats.total_files} files processed, "
                f"{stats.directories_processed} directories, "
                f"{files_per_sec:.1f} files/sec, {mb_per_sec:.1f} MB/sec"
            )
            
            if current_file:
                self.logger.debug(f"Currently processing: {current_file}")
            
            self.last_update = current_time


def setup_logging(log_level: str = "INFO", log_file: str = "directory_search.log") -> logging.Logger:
    """Set up optimized logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler with limited output for performance
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    return logger


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def main():
    """Optimized main function for large directory processing"""
    parser = argparse.ArgumentParser(
        description="Optimized directory tree search for large directories (1TB+)"
    )
    parser.add_argument("directory", type=str, help="Directory path to search in")
    parser.add_argument("keyword", type=str, help="Keyword to search for in files")
    parser.add_argument("--case-sensitive", action="store_true", help="Case-sensitive search")
    parser.add_argument("--include", nargs="+", help="File patterns to include (e.g., *.py *.txt)")
    parser.add_argument("--exclude-dirs", nargs="+", help="Directory patterns to exclude")
    parser.add_argument("--max-workers", type=int, help="Maximum worker threads")
    parser.add_argument("--max-depth", type=int, default=50, help="Maximum directory depth")
    parser.add_argument("--chunk-size", type=int, default=100, help="File processing chunk size")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--stats-only", action="store_true", help="Show only statistics")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Validate directory
        directory_path = Path(args.directory).resolve()
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Invalid directory: {directory_path}")
            sys.exit(1)
        
        logger.info(f"Starting optimized search in: {directory_path}")
        logger.info(f"Searching for: '{args.keyword}' (case-sensitive: {args.case_sensitive})")
        logger.info(f"Available CPU cores: {os.cpu_count()}")
        logger.info(f"Available memory: {format_size(psutil.virtual_memory().total)}")
        
        # Initialize components
        tree_builder = OptimizedTreeBuilder(logger, args.max_depth)
        searcher = ParallelKeywordSearcher(
            logger, args.case_sensitive, args.max_workers, args.chunk_size
        )
        progress_monitor = ProgressMonitor(logger)
        
        # Convert include patterns to tuple for caching
        include_patterns = tuple(args.include) if args.include else None
        
        # Start search
        start_time = time.time()
        
        # Generate file list and search in parallel
        file_generator = tree_builder.scan_directory_fast(
            directory_path, include_patterns, args.exclude_dirs
        )
        
        matching_files, search_stats = searcher.search_files_parallel(file_generator, args.keyword)
        
        # Final statistics
        total_time = time.time() - start_time
        total_stats = tree_builder.stats
        
        # Display results
        print("\n" + "="*60)
        print("SEARCH RESULTS")
        print("="*60)
        
        if not args.stats_only and matching_files:
            print(f"Found {len(matching_files)} file(s) containing '{args.keyword}':")
            for file_path in matching_files:
                print(f"  • {file_path}")
        elif not matching_files:
            print(f"No files found containing '{args.keyword}'")
        
        # Performance statistics
        print(f"\nPERFORMANCE STATISTICS:")
        print(f"  Total runtime: {total_time:.2f} seconds")
        print(f"  Files processed: {total_stats.total_files:,}")
        print(f"  Directories scanned: {total_stats.directories_processed:,}")
        print(f"  Files matched: {search_stats.files_matched:,}")
        print(f"  Files skipped: {search_stats.files_skipped:,}")
        print(f"  Files with errors: {search_stats.files_error:,}")
        print(f"  Data processed: {format_size(search_stats.bytes_processed)}")
        print(f"  Processing rate: {total_stats.total_files/total_time:.1f} files/sec")
        print(f"  Throughput: {format_size(search_stats.bytes_processed/total_time)}/sec")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"Search results for '{args.keyword}' in {directory_path}\n")
                f.write(f"Generated on {time.ctime()}\n\n")
                for file_path in matching_files:
                    f.write(f"{file_path}\n")
            logger.info(f"Results saved to: {args.output}")
        
        logger.info("Search completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
