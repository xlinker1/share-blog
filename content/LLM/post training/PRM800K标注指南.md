---
draft: false
tags:
---

很重要但也容易被忽视的地方。数据标注决定了训练出来的PRM到底是啥。标注指南也相当于对人类的prompt，想要扩大标注数据集的(数量/领域)，最直接的办法就是将它转换为对ai的prompt。因此绝对应该仔细阅读。
指令见[prm800k/prm800k/instructions at main · openai/prm800k · GitHub](https://github.com/openai/prm800k/tree/main/prm800k/instructions)

下面是prm800k/instructions_phase_2.pdf，对思考过程中good okey bad的描述还是很细的。

![[Pasted image 20241028141750.png]]

![[Pasted image 20241029161139.png]]

![[Pasted image 20241029161430.png]]
错了之后还要给问题解决的进度评分，点评一些ai写的错误解释，如果都不行就自己写一下哪里错了，还要对可能的后续步骤打分

![[Pasted image 20241029161715.png]]

## Great

A **Great** option is all of the following:
- Correct
	- Nothing it states is wrong
- Verifiable
	- It should take you no more than about 30 seconds to verify that the statement is correct (more if you are rusty with the problem's general area, like trigonometry or number theory - use your judgment)
	- You might need to use a calculator to check some simple calculations, but if you feel like you need to take out paper and pencil to check that the suggestion is right, mark it **Bad** because it wasn't explained well enough 
	- You might also need to look up a theorem (e.g. a trig identity) in order to verify correctness. lf you can do this with a quick Google search you can mark it **Great** , if it's more obscure than that mark it **Bad**
- Appropriate
	- It fits correctly into the context of the previous steps
	- If the previous steps contain a mistake that wasn't noticed before, it's ok for a **Great** option to point out the mistake
- Insightful
	- They are reasonable things a smart human might try while solving the problem. 
	- Even if it's ultimately the wrong direction (but not immediately obviously a wrong direction), it can still be insightful to try. We want to teach the Al to learn how to recover from trying something that doesn't work out right away!
	- If the option is simply restating one of the previous steps, without adding any additional insight or setting direction for what to do next, mark it **Okay**
	- If the option is a statement of encouragement ('Good job!') but doesn't offer any additional insight or setting direction for what to do next, mark it **Okay**
	- Sometimes the suggestion might add the tiniest amount of further insight or .m.,m,m,.guides the solution forward in a particular direction only slightly - it can be marked **Great** or **Okay** depending on context or even other suggestions that you see
- If the option has a final answer, it should not only be correct, but also clearly follow from the rest of the reasoning. Don't select options with incorrect final answers!（如果step里包含最终答案，不仅需要正确，也需要能清晰的从上文的推理中得出。）

## Okey

Okay options sound like something a person would say, they just don't contribute anything of essence to the conversation. They're reasonable, verifiably correct, and appropriate, but they're also redundant, stalling, or just don't add any value.（冗余，拖延，或者根本无法带来任何价值）

For example they might just repeat a fact or the problem itself, provide some encouragement without furthering the conversation ("Great job!"), complain that the problem is hard or say that it's easy, etc.

Another kind of correct statement that should be marked Okay is one that makes progress along a direction, but it'mhnms *stalling* on making a more decisive amount of progress. For example if the problem is to find the last digit of 2^10000, the first couple of steps that look like
2^1 = 2, ends in 2
2^2 = 4, ends in 4
2^3=8, ends in 8,
2^4=16, ends in 6,
2^5=32,ends in 2,
2^6=64, ends in 4,
2^7=128, ends in 8,
2^8=256, ends in 6,
2^9=512, ends in 2,
can all be marked **Great** because they are contributing to our understanding of the problem, but if this continues for too long, eventually it's just stalling on making the critical observation that there is a pattern in the last digits. So at some point (which might
reasonably be anywhere between 2^6 and 2^10) please stop marking the suggestions as **Great** and instead mark it as **okay**.

## Bad

Any of these characteristics will make an option **Bad**：
- Hard to verify
	- It's not explained well and you'd need to use paper and pencil to check that it's correct
- Wrong
	- Even if most of the suggestion is correct, but it also states something that is wrong, mark it Bad
- Contains gibberish
- Contains off-topic text or non-sequiturs
- Suggests attempting something that is unreasonable for this problem
- Derails(偏离) the conversation
- Leads the solution into an immediately obvious dead end or makes it go in circles
- Leads the solution into a repetitious pattern that should obviously be stopped
- Refers to an external link that it claims the solution relies on (e.g. a link to a graph or image)-please don't click on computer-generated external links
- Refers to a graph or picture that is not included, and the solution relies on it (i.e. it's hard to imagine what the graph or picture is supposed to be unambiguously)

## Unsure

Anything that the instructions didn't cover you can mark as "Unsure" and then move on. Please use this sparingly.

Sometimes you might be uncertain if something qualifies as **Great** or merely **Okay**. Feel free to exercise your own judgment. For example if the problem is about n!, a statement “n!=1\*2\*...\*n" might be argued to be **Great** because the first step towards a solution is remembering what n! is, or **Okay** because everyone knows what n! is, so it's just restating
the obvious.

Similarly the distinction between **Okay** and **Bad** might sometimes be fuzzy. For example "Hey, listen, this was great!" could be "okay" if you think the tone is appropriate, or "bad" if you think it's weird to use this tone while discussing math problems.

In both cases, just pick one of the **Great**, **Okay**, or **Bad** options if you find them appropriate and justifiable. You don't have to be 100% systematic about these gray areas.

Pick **unsure** if you encounter a statement that doesn't really satisfy any of the criteria covered above for **Great**, **Okay**, or **Bad**. We will review such statements and update the instructions accordingly.

## Rating the final answer
When the model outputs "Answer: ..." we regard it as the end of the solution. If the model didn't make a mistake until that point, this is the step where we can check if the solution is complete. If it is not complete, mark this last step as incorrect even if the numerical answer
当模型输出“Answer：…”时，我们将其视为解决方案的结束。如果模型在此之前没有出错，这一步我们可以检查解是否完整。如果不完整，即使是数字答案，也将最后一步标记为不正确

Problem:
Solve x^2+x=0.
Solution:
- The equation is equivalent to x(x+1)=0 -> "good"
- If x is not 0, then we can divide by x to get x+1=0 -> "good" (nothing wrong is stated) or maybe "okay" (because the step creates a little confusion about what to do if x is equal to 0)
- Answer: x= - 1 -> "bad" (because the solution is incomplete, the case x=0 wasn't considered)


For a question that requires a proof, a correct answer might be just "QED", "done", "now we proved the original statement", or something similar. Similar to numeric questions, don't mark the last step as "good" unless the full solution is complete.
对于一个需要证明的问题，正确的答案可能只是“QED”，“done”，“now we prove the original statement”，或者类似的东西。与数字问题类似，除非完整的解决方案完成，否则不要将最后一步标记为“好”。

Solution:
- The equation is equivalent to x(x+1)=0 -> "good"
- If x is not 0, then we can divide by x to get x+1=0 -> "good" or maybe "okay"
- Since x=-1 is an integer we are done
  Answer: QED -> "bad" (because the solution is incomplete, the case x=0 wasn't cosidered explicitly)

On the other hand this solution will be completely “good":
Solution:
- The equation is equivalent to x(x+1)=0 ->“good"
- lf x is not 0, then we can divide byx to get x+1=0 ->“good"
- Since x=-1 is an integer. The remaining case x=0 is an integer too, so we are done
  Answer: QED ->“good"


- Reporting a bad problem
	- When is a problem statement bad 
- When to Give Up on a Conversation
- Focus on substance, not nitpicks - examples