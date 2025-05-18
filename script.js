document.addEventListener('DOMContentLoaded', () => {
    const questionsListDiv = document.getElementById('questions-list');
    const hintModal = document.getElementById('hint-modal');
    const hintQuestionTitle = document.getElementById('hint-question');
    const hintContentDiv = document.getElementById('hint-content');
    const closeBtn = document.querySelector('.close-btn');

    // Complete list of Fasal coding questions with hints and answers
    const fasalQuestions = [
        // ... (your existing question array remains the same)
                   {
    "question": "Find the longest substring without repeating characters.",
    "description": "Tests your understanding of sliding window and hash map.",
    "hint": "Use two pointers and a set to track characters.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function lengthOfLongestSubstring(s) {
  let set = new Set();
  let left = 0, maxLength = 0;

  for (let right = 0; right < s.length; right++) {
    while (set.has(s[right])) {
      set.delete(s[left]);
      left++;
    }
    set.add(s[right]);
    maxLength = Math.max(maxLength, right - left + 1);
  }
  return maxLength;
}
      </code></pre>
    `
  },
  {
    "question": "Check if a linked list is a palindrome.",
    "description": "Validates understanding of two-pointer and reversal.",
    "hint": "Reverse the second half and compare with the first.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function isPalindrome(head) {
  let slow = head, fast = head, prev = null;

  while (fast && fast.next) {
    fast = fast.next.next;
    [prev, slow.next, slow] = [slow, prev, slow.next];
  }

  if (fast) slow = slow.next;

  while (slow && prev && slow.val === prev.val) {
    slow = slow.next;
    prev = prev.next;
  }

  return !slow;
}
      </code></pre>
    `
  },
  {
    "question": "Implement a LRU (Least Recently Used) cache.",
    "description": "Checks your ability to use hash maps and doubly linked lists.",
    "hint": "Combine a hash map and a doubly linked list.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key) {
    if (!this.cache.has(key)) return -1;
    const val = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, val);
    return val;
  }

  put(key, value) {
    if (this.cache.has(key)) this.cache.delete(key);
    this.cache.set(key, value);
    if (this.cache.size > this.capacity) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }
}
      </code></pre>
    `
  },
  {
    "question": "Merge two sorted linked lists.",
    "description": "Tests recursion and list traversal logic.",
    "hint": "Compare head nodes and merge recursively.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function mergeTwoLists(l1, l2) {
  if (!l1 || !l2) return l1 || l2;
  if (l1.val < l2.val) {
    l1.next = mergeTwoLists(l1.next, l2);
    return l1;
  } else {
    l2.next = mergeTwoLists(l1, l2.next);
    return l2;
  }
}
      </code></pre>
    `
  },
  {
    "question": "Detect cycle in a directed graph.",
    "description": "Checks your DFS and visited state management skills.",
    "hint": "Use DFS with recursion stack tracking.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function hasCycle(graph) {
  const visited = new Set();
  const recStack = new Set();

  function dfs(node) {
    if (recStack.has(node)) return true;
    if (visited.has(node)) return false;

    visited.add(node);
    recStack.add(node);

    for (let neighbor of graph[node] || []) {
      if (dfs(neighbor)) return true;
    }

    recStack.delete(node);
    return false;
  }

  for (let node in graph) {
    if (dfs(node)) return true;
  }

  return false;
}
      </code></pre>
    `
  },
  {
    "question": "Find the minimum in a rotated sorted array.",
    "description": "Binary search twist to identify rotation.",
    "hint": "Modify binary search based on middle value.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function findMin(nums) {
  let left = 0, right = nums.length - 1;

  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (nums[mid] > nums[right]) left = mid + 1;
    else right = mid;
  }

  return nums[left];
}
      </code></pre>
    `
  },
  {
    "question": "Count the number of islands in a 2D grid.",
    "description": "Tests DFS/BFS and matrix traversal.",
    "hint": "Use DFS to mark connected land parts.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function numIslands(grid) {
  let count = 0;

  function dfs(i, j) {
    if (
      i < 0 || i >= grid.length || 
      j < 0 || j >= grid[0].length || 
      grid[i][j] === '0'
    ) return;
    grid[i][j] = '0';
    dfs(i+1, j);
    dfs(i-1, j);
    dfs(i, j+1);
    dfs(i, j-1);
  }

  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[0].length; j++) {
      if (grid[i][j] === '1') {
        count++;
        dfs(i, j);
      }
    }
  }

  return count;
}
      </code></pre>
    `
  },
  {
    "question": "Check if a number is power of two.",
    "description": "Checks knowledge of bitwise operations.",
    "hint": "n & (n - 1) will be zero only for powers of 2.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}
      </code></pre>
    `
  },
  {
    "question": "Reverse nodes in k-group in a linked list.",
    "description": "Advanced linked list manipulation.",
    "hint": "Reverse each group of k nodes iteratively or recursively.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">function reverseKGroup(head, k) {
  let count = 0, ptr = head;
  while (count < k && ptr) {
    ptr = ptr.next;
    count++;
  }

  if (count === k) {
    let prev = reverseKGroup(ptr, k);
    while (count-- > 0) {
      let temp = head.next;
      head.next = prev;
      prev = head;
      head = temp;
    }
    return prev;
  }
  return head;
}
      </code></pre>
    `
  },
  {
    "question": "Implement queue using stacks.",
    "description": "Classic data structure transformation.",
    "hint": "Use two stacks: input and output.",
    "answer": `
      <p><strong>Sample Answer (JavaScript):</strong></p>
      <pre><code class="language-javascript">class MyQueue {
  constructor() {
    this.input = [];
    this.output = [];
  }

  push(x) {
    this.input.push(x);
  }

  pop() {
    if (!this.output.length) {
      while (this.input.length) {
        this.output.push(this.input.pop());
      }
    }
    return this.output.pop();
  }

  peek() {
    if (!this.output.length) {
      while (this.input.length) {
        this.output.push(this.input.pop());
      }
    }
    return this.output[this.output.length - 1];
  }

  empty() {
    return !this.input.length && !this.output.length;
  }
}
      </code></pre>
    `
  },
  {
  "question": "Reverse a singly linked list.",
  "description": "Tests understanding of linked list manipulation.",
  "hint": "Use three pointers: prev, current, and next.",
  "answer": `
    <pre><code class="language-java">class ListNode {
  int val;
  ListNode next;
  ListNode(int x) { val = x; }
}

public class ReverseLinkedList {
  public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    while (head != null) {
      ListNode next = head.next;
      head.next = prev;
      prev = head;
      head = next;
    }
    return prev;
  }
}
    </code></pre>
  `
},
{
  "question": "Find the length of the longest substring without repeating characters.",
  "description": "Checks knowledge of sliding window technique.",
  "hint": "Use HashSet and two pointers.",
  "answer": `
    <pre><code class="language-java">import java.util.HashSet;

public class LongestSubstring {
  public int lengthOfLongestSubstring(String s) {
    int max = 0, left = 0;
    HashSet<Character> set = new HashSet<>();
    for (int right = 0; right < s.length(); right++) {
      while (set.contains(s.charAt(right))) {
        set.remove(s.charAt(left++));
      }
      set.add(s.charAt(right));
      max = Math.max(max, right - left + 1);
    }
    return max;
  }
}
    </code></pre>
  `
},
{
  "question": "Merge two sorted linked lists into one sorted list.",
  "description": "Tests recursive thinking and list manipulation.",
  "hint": "Use recursion or iteration by comparing node values.",
  "answer": `
    <pre><code class="language-java">class ListNode {
  int val;
  ListNode next;
  ListNode(int x) { val = x; }
}

public class MergeSortedLists {
  public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null) return l2;
    if (l2 == null) return l1;
    if (l1.val < l2.val) {
      l1.next = mergeTwoLists(l1.next, l2);
      return l1;
    } else {
      l2.next = mergeTwoLists(l1, l2.next);
      return l2;
    }
  }
}
    </code></pre>
  `
},
{
  "question": "Check if the input string has valid parentheses.",
  "description": "Tests use of stacks and basic string logic.",
  "hint": "Use a stack to match opening and closing brackets.",
  "answer": `
    <pre><code class="language-java">import java.util.Stack;

public class ValidParentheses {
  public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    for (char c : s.toCharArray()) {
      if (c == '(' || c == '{' || c == '[') {
        stack.push(c);
      } else {
        if (stack.isEmpty()) return false;
        char top = stack.pop();
        if ((c == ')' && top != '(') ||
            (c == '}' && top != '{') ||
            (c == ']' && top != '[')) return false;
      }
    }
    return stack.isEmpty();
  }
}
    </code></pre>
  `
},
{
  "question": "Detect if a linked list has a cycle.",
  "description": "Tests knowledge of Floyd’s cycle detection algorithm.",
  "hint": "Use two pointers: slow and fast.",
  "answer": `
    <pre><code class="language-java">class ListNode {
  int val;
  ListNode next;
  ListNode(int x) { val = x; next = null; }
}

public class LinkedListCycle {
  public boolean hasCycle(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
      slow = slow.next;
      fast = fast.next.next;
      if (slow == fast) return true;
    }
    return false;
  }
}
    </code></pre>
  `
},
{
  "question": "Return the k most frequent elements from the array.",
  "description": "Tests use of HashMap and PriorityQueue.",
  "hint": "Use a max heap or bucket sort.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class TopKFrequent {
  public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> count = new HashMap<>();
    for (int n : nums) count.put(n, count.getOrDefault(n, 0) + 1);
    
    PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> count.get(b) - count.get(a));
    heap.addAll(count.keySet());

    int[] result = new int[k];
    for (int i = 0; i < k; i++) {
      result[i] = heap.poll();
    }
    return result;
  }
}
    </code></pre>
  `
},
{
  "question": "Implement a stack using two queues.",
  "description": "Tests creative implementation with queues.",
  "hint": "Use two queues to simulate stack behavior (LIFO).",
  "answer": `
    <pre><code class="language-java">import java.util.LinkedList;
import java.util.Queue;

class MyStack {
  Queue<Integer> q1 = new LinkedList<>();
  Queue<Integer> q2 = new LinkedList<>();

  public void push(int x) {
    q2.add(x);
    while (!q1.isEmpty()) {
      q2.add(q1.remove());
    }
    Queue<Integer> temp = q1;
    q1 = q2;
    q2 = temp;
  }

  public int pop() {
    return q1.remove();
  }

  public int top() {
    return q1.peek();
  }

  public boolean empty() {
    return q1.isEmpty();
  }
}
    </code></pre>
  `
},
{
  "question": "Return an array where each element is the product of all the other elements.",
  "description": "Tests understanding of array manipulation without using division.",
  "hint": "Use prefix and suffix product arrays.",
  "answer": `
    <pre><code class="language-java">public class ProductArray {
  public int[] productExceptSelf(int[] nums) {
    int[] res = new int[nums.length];
    int left = 1, right = 1;
    
    for (int i = 0; i < nums.length; i++) {
      res[i] = left;
      left *= nums[i];
    }
    
    for (int i = nums.length - 1; i >= 0; i--) {
      res[i] *= right;
      right *= nums[i];
    }
    
    return res;
  }
}
    </code></pre>
  `
},
{
  "question": "Count the number of islands in a 2D grid (1's as land, 0's as water).",
  "description": "Tests DFS/BFS and matrix traversal logic.",
  "hint": "Use DFS to mark all connected land.",
  "answer": `
    <pre><code class="language-java">public class NumberOfIslands {
  public int numIslands(char[][] grid) {
    int count = 0;
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] == '1') {
          dfs(grid, i, j);
          count++;
        }
      }
    }
    return count;
  }

  private void dfs(char[][] grid, int i, int j) {
    if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return;
    grid[i][j] = '0';
    dfs(grid, i + 1, j);
    dfs(grid, i - 1, j);
    dfs(grid, i, j + 1);
    dfs(grid, i, j - 1);
  }
}
    </code></pre>
  `
},
{
  "question": "Find the first non-repeating character in a string.",
  "description": "Tests string manipulation and use of data structures like hash maps.",
  "hint": "Use a LinkedHashMap to track character counts while preserving order.",
  "answer": `
    <p><strong>Sample Answer (Java):</strong></p>
    <pre><code class="language-java">import java.util.*;

public class FirstUniqueChar {
    public static char firstUniqChar(String s) {
        Map&lt;Character, Integer&gt; count = new LinkedHashMap&lt;&gt;();
        for (char c : s.toCharArray()) {
            count.put(c, count.getOrDefault(c, 0) + 1);
        }
        for (char c : count.keySet()) {
            if (count.get(c) == 1) return c;
        }
        return '_';
    }
}
    </code></pre>
  `
},
{
  "question": "Check if a number is a power of 4.",
  "description": "Tests understanding of mathematical patterns and bitwise operations.",
  "hint": "A power of 4 has only one bit set and that bit is in the right position.",
  "answer": `
    <pre><code class="language-java">public class PowerOfFour {
    public boolean isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0xAAAAAAAA) == 0;
    }
}
    </code></pre>
  `
},
{
  "question": "Convert Roman numeral to integer.",
  "description": "Assesses understanding of string parsing and control logic.",
  "hint": "Process the string from left to right while checking special cases.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class RomanToInteger {
    public int romanToInt(String s) {
        Map&lt;Character, Integer&gt; map = Map.of(
            'I', 1, 'V', 5, 'X', 10, 'L', 50,
            'C', 100, 'D', 500, 'M', 1000);
        int total = 0;
        for (int i = 0; i &lt; s.length(); i++) {
            int val = map.get(s.charAt(i));
            if (i + 1 &lt; s.length() && val &lt; map.get(s.charAt(i + 1))) {
                total -= val;
            } else {
                total += val;
            }
        }
        return total;
    }
}
    </code></pre>
  `
},
{
  "question": "Detect a cycle in a linked list.",
  "description": "Common question testing Floyd’s Cycle detection algorithm.",
  "hint": "Use slow and fast pointers to detect the cycle.",
  "answer": `
    <pre><code class="language-java">class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; next = null; }
}

public class CycleDetection {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            if (slow == fast) return true;
            slow = slow.next;
            fast = fast.next.next;
        }
        return false;
    }
}
    </code></pre>
  `
},
{
  "question": "Group anagrams together from a list of strings.",
  "description": "Requires knowledge of hashing and sorting strings.",
  "hint": "Sort each word and use it as a key in a hash map.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class GroupAnagrams {
    public List&lt;List&lt;String&gt;&gt; groupAnagrams(String[] strs) {
        Map&lt;String, List&lt;String&gt;&gt; map = new HashMap&lt;&gt;();
        for (String s : strs) {
            char[] arr = s.toCharArray();
            Arrays.sort(arr);
            String key = new String(arr);
            map.computeIfAbsent(key, k -&gt; new ArrayList&lt;&gt;()).add(s);
        }
        return new ArrayList&lt;&gt;(map.values());
    }
}
    </code></pre>
  `
},
{
  "question": "Find the longest substring without repeating characters.",
  "description": "Tests sliding window technique.",
  "hint": "Use a HashSet to track characters in the current window.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class LongestUniqueSubstring {
    public int lengthOfLongestSubstring(String s) {
        Set&lt;Character&gt; set = new HashSet&lt;&gt;();
        int max = 0, left = 0;
        for (int right = 0; right &lt; s.length(); right++) {
            while (!set.add(s.charAt(right))) {
                set.remove(s.charAt(left++));
            }
            max = Math.max(max, right - left + 1);
        }
        return max;
    }
}
    </code></pre>
  `
},
{
  "question": "Find all duplicates in an array where 1 ≤ a[i] ≤ n (n = size of array).",
  "description": "Tests ability to solve in O(n) time and no extra space.",
  "hint": "Use index marking by negating values.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class FindDuplicates {
    public List&lt;Integer&gt; findDuplicates(int[] nums) {
        List&lt;Integer&gt; res = new ArrayList&lt;&gt;();
        for (int i = 0; i &lt; nums.length; i++) {
            int index = Math.abs(nums[i]) - 1;
            if (nums[index] &lt; 0) res.add(index + 1);
            nums[index] = -nums[index];
        }
        return res;
    }
}
    </code></pre>
  `
},
{
  "question": "Implement a stack using two queues.",
  "description": "Good test of data structure transformation and queue operations.",
  "hint": "Use one queue for push, and the other for pop by rotating elements.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class MyStack {
    Queue&lt;Integer&gt; q1 = new LinkedList&lt;&gt;();
    Queue&lt;Integer&gt; q2 = new LinkedList&lt;&gt;();

    public void push(int x) {
        q2.add(x);
        while (!q1.isEmpty()) {
            q2.add(q1.remove());
        }
        Queue&lt;Integer&gt; temp = q1;
        q1 = q2;
        q2 = temp;
    }

    public int pop() {
        return q1.remove();
    }

    public int top() {
        return q1.peek();
    }

    public boolean empty() {
        return q1.isEmpty();
    }
}
    </code></pre>
  `
},
{
  "question": "Generate all valid parentheses combinations for n pairs.",
  "description": "Classic backtracking problem involving recursion.",
  "hint": "Track counts of open and close parentheses used.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class GenerateParentheses {
    public List&lt;String&gt; generateParenthesis(int n) {
        List&lt;String&gt; result = new ArrayList&lt;&gt;();
        backtrack(result, "", 0, 0, n);
        return result;
    }

    private void backtrack(List&lt;String&gt; result, String current, int open, int close, int max) {
        if (current.length() == max * 2) {
            result.add(current);
            return;
        }
        if (open &lt; max) backtrack(result, current + "(", open + 1, close, max);
        if (close &lt; open) backtrack(result, current + ")", open, close + 1, max);
    }
}
    </code></pre>
  `
},
{
  "question": "Find the kth largest element in an array.",
  "description": "Tests knowledge of heaps and partitioning.",
  "hint": "Use a min-heap of size k.",
  "answer": `
    <pre><code class="language-java">import java.util.*;

public class KthLargest {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue&lt;Integer&gt; minHeap = new PriorityQueue&lt;&gt;();
        for (int num : nums) {
            minHeap.add(num);
            if (minHeap.size() &gt; k) minHeap.poll();
        }
        return minHeap.peek();
    }
}
    </code></pre>
  `
},
{
  "question": "What unique perspective or skill would you bring to LinkedIn?",
  "description": "Evaluates how your background adds diversity and value.",
  "hint": "Mention technical skills, past experiences, or viewpoints that enhance team strength.",
  "answer": "Coming from a rural background, I understand the challenges faced by underrepresented users. I aim to create inclusive digital solutions that are user-friendly and help bridge the opportunity gap."
},
{
  "question": "Describe a time you failed and what you learned from it.",
  "description": "Assesses resilience, growth mindset, and self-awareness.",
  "hint": "Be honest and focus on what you learned and how you improved.",
  "answer": "During an internship, I underestimated the time needed to debug a module. It delayed team progress. I learned to better estimate time, test early, and ask for help when needed. It made me a more reliable teammate."
},
{
  "question": "Why do you want to join LinkedIn specifically?",
  "description": "Tests your knowledge of the company’s products, culture, and mission.",
  "hint": "Mention LinkedIn’s impact, innovation, and your alignment with their values like 'Members First' and 'Act Like an Owner'.",
  "answer": "LinkedIn inspires me because of its purpose-driven platform and commitment to professional growth. I admire how it blends social connectivity with career advancement, and I want to be part of that impact."
},
{
  "question": "Can you tell me about a time you solved a conflict within a team?",
  "description": "Evaluates interpersonal skills and emotional intelligence.",
  "hint": "Use the STAR (Situation, Task, Action, Result) method to clearly communicate how you handled it.",
  "answer": "In a college project, two teammates had differing priorities. I initiated a neutral discussion, clarified shared goals, and divided tasks based on strengths. This improved collaboration and we successfully delivered ahead of time."
},
{
  "question": "How do you see yourself contributing to LinkedIn’s vision of 'creating economic opportunity for every member of the global workforce'?",
  "description": "Tests alignment with company values and vision.",
  "hint": "Frame your answer with examples of how you want to help users find better job opportunities, learn new skills, or build meaningful connections.",
  "answer": "I align with this vision by aiming to create intuitive, inclusive solutions that help professionals connect with opportunities. Whether through optimizing algorithms or building accessible features, I’d ensure all members have a fair shot at success."
},
{
  "question": "What does 'transformation at scale' mean to you, and how can you contribute to it at LinkedIn?",
  "description": "Assesses your understanding of LinkedIn's mission and your alignment with their large-scale impact goals.",
  "hint": "Talk about your ability to build solutions that scale and help empower professionals globally.",
  "answer": "‘Transformation at scale’ means creating impactful solutions that benefit a massive number of users. I can contribute by building efficient, user-centric software that improves engagement, accessibility, or career outcomes for LinkedIn’s members."
}
    ];

    fasalQuestions.forEach((question, index) => {
        const questionDiv = document.createElement('div');
        questionDiv.classList.add('question-item');

        const title = document.createElement('h3');
        title.textContent = `${index + 1}. ${question.question}`;

        const description = document.createElement('p');
        description.textContent = question.description;

        // Create button container
        const buttonContainer = document.createElement('div');
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '10px';
        buttonContainer.style.marginTop = '15px';

        // Hint Button
        const hintButton = document.createElement('button');
        hintButton.textContent = 'Show Hint';
        hintButton.style.padding = '10px 20px';
        hintButton.style.border = 'none';
        hintButton.style.borderRadius = '5px';
        hintButton.style.backgroundColor = '#4CAF50';
        hintButton.style.color = 'white';
        hintButton.style.fontWeight = 'bold';
        hintButton.style.cursor = 'pointer';
        hintButton.style.transition = 'all 0.3s ease';
        hintButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Hover effect for hint button
        hintButton.addEventListener('mouseover', () => {
            hintButton.style.backgroundColor = '#45a049';
            hintButton.style.transform = 'translateY(-2px)';
            hintButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        hintButton.addEventListener('mouseout', () => {
            hintButton.style.backgroundColor = '#4CAF50';
            hintButton.style.transform = 'translateY(0)';
            hintButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        hintButton.addEventListener('click', () => {
            hintQuestionTitle.textContent = question.question;
            hintContentDiv.innerHTML = `<p>${question.hint}</p>`;
            hintModal.style.display = 'block';
        });

        // Answer Button
        const answerButton = document.createElement('button');
        answerButton.textContent = 'Show Answer';
        answerButton.style.padding = '10px 20px';
        answerButton.style.border = 'none';
        answerButton.style.borderRadius = '5px';
        answerButton.style.backgroundColor = '#2196F3';
        answerButton.style.color = 'white';
        answerButton.style.fontWeight = 'bold';
        answerButton.style.cursor = 'pointer';
        answerButton.style.transition = 'all 0.3s ease';
        answerButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Hover effect for answer button
        answerButton.addEventListener('mouseover', () => {
            answerButton.style.backgroundColor = '#0b7dda';
            answerButton.style.transform = 'translateY(-2px)';
            answerButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        answerButton.addEventListener('mouseout', () => {
            answerButton.style.backgroundColor = '#2196F3';
            answerButton.style.transform = 'translateY(0)';
            answerButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        answerButton.addEventListener('click', () => {
            hintQuestionTitle.textContent = question.question;
            hintContentDiv.innerHTML = question.answer;
            hintModal.style.display = 'block';
        });

        // Add buttons to container
        buttonContainer.appendChild(hintButton);
        buttonContainer.appendChild(answerButton);

        questionDiv.appendChild(title);
        questionDiv.appendChild(description);
        questionDiv.appendChild(buttonContainer);
        questionsListDiv.appendChild(questionDiv);
    });

    closeBtn.addEventListener('click', () => {
        hintModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === hintModal) {
            hintModal.style.display = 'none';
        }
    });
});