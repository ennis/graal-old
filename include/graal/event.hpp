#pragma once
#include <memory>

namespace graal {
	
	namespace detail {
		class queue_impl;

		class event_impl {
		public:
			
		private:
			std::shared_ptr<queue_impl> queue_;
		};
	}



	class event {
	public:
		enum class status {
			// enqueued but not submitted to opengl
			enqueued,
			// submitted to opengl
			submitted,
			
		};



		void wait() { impl_->wait(); }


	private:
		std::shared_ptr<event_impl> impl_;
	};

}